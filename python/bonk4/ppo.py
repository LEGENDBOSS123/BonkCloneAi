"""ppo4 PPO: feedforward actor over the 18 joint actions + an SPPO-style
prefix-GRU outcome critic (see critic.py).

The two halves train on DIFFERENT data layouts, which is the point of the
split:

  actor  — flat rows [N, D]. Order does not matter, so it gets everything:
           the learner seat, the opponent seat, and mirror augmentation.
  critic — sequences [T, E]. Order is everything (h_t must summarise the
           prefix), so it trains per-env-column with truncated BPTT.

There is no GAE and no bootstrapping. The critic is trained by cross-entropy
against the episode's real outcome, and the advantage is the Monte-Carlo
residual A_t = R - V(prefix_t).
"""

import numpy as np
import torch
import torch.nn.functional as F

from bonk.networks import MLP
from .actor import ActorNet

from . import config as C
from .critic import PrefixCritic
from .hindsight import HindsightBaseline


def entropy_coef_at(episode: int) -> float:
    """Anneal START -> END over ENTROPY_DECAY_EPISODES, then hold at END.
    "linear": constant absolute decrease per episode.
    "exponential": constant RATIO per episode — drops fast early, long low
    tail. Reaches END exactly at ENTROPY_DECAY_EPISODES either way."""
    frac = min(1.0, episode / C.ENTROPY_DECAY_EPISODES)
    if getattr(C, "ENTROPY_DECAY_MODE", "linear") == "exponential":
        return C.ENTROPY_COEF_START * (C.ENTROPY_COEF_END / C.ENTROPY_COEF_START) ** frac
    return C.ENTROPY_COEF_START + (C.ENTROPY_COEF_END - C.ENTROPY_COEF_START) * frac


def _critic_seqs_per_mb(B: int, T: int) -> int:
    """Sequences per critic minibatch, snapped to a DIVISOR of B.

    Two things matter here and they pull against each other:

    * Total GRU work is epochs*T*B however it is split, but the unroll is a
      Python loop of length T per minibatch, so the dispatch overhead is
      (B/seqs_per_mb)*T. Bigger minibatches => fewer passes over that loop.
      Measured 6.3s -> 4.4s on one update just from this.
    * Only FULL minibatches run (a ragged tail is a new tensor shape, and
      torch-MPS caches a compiled graph per shape forever). So any
      seqs_per_mb that does not divide B DISCARDS up to seqs_per_mb-1
      sequences every epoch — at seqs_per_mb=1000 with B=5120 that was 120
      columns, silently, every epoch.

    Snapping to a divisor gets the large-minibatch speedup with zero loss.
    """
    if C.CRITIC_SEQS_PER_MB:
        want = C.CRITIC_SEQS_PER_MB
    else:
        # Fit the BPTT working set inside the budget. ~6 stored tensors per
        # GRUCell step per layer (gate pre-activations + outputs), 4 bytes each.
        # memory follows the BPTT chunk, not the full window
        depth = min(T, C.CRITIC_BPTT_CHUNK) if C.CRITIC_BPTT_CHUNK else T
        # ~12 stored tensors per GRUCell step per layer (3 gate pre-acts,
        # their activations, the candidate, the blend, plus the encoder's
        # Linear/LayerNorm/ReLU saves). The old estimate of 6 under-counted
        # by ~2x, which is how a '1400 MB' budget became a 3.3 GB OOM.
        per_seq = max(1.0, depth * C.CRITIC_HIDDEN * 4 * C.CRITIC_LAYERS * 12)
        want = int(C.CRITIC_BPTT_BUDGET_MB * 1e6 / per_seq)
    want = max(1, want)
    want = min(B, want)
    n_mb = max(1, -(-B // want))          # ceil(B / want)
    while n_mb < B and B % n_mb:          # smallest divisor >= n_mb
        n_mb += 1
    return max(1, B // n_mb)


class PPOAgent:
    def __init__(self, state_dim: int, num_actions: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        # Plain MLP: what the league freezes, what export_model ships and what
        # play3.mjs rebuilds. Keeping it an MLP is what lets the critic change
        # shape freely without touching the deploy path.
        # actor.trunk IS the shared encoder (34 -> 256 -> 256); actor.out is the
        # policy head. Keeping the pair inside one MLP is what lets export_model
        # and play3.mjs stay untouched — on disk it is still MLP(34,[256,256],18).
        self.actor = ActorNet(state_dim, num_actions).to(self.device)
        if C.MOVE_TRANSPLANT:
            import os
            pth = os.path.join(os.path.dirname(__file__), C.MOVE_CORE_PATH)
            if os.path.exists(pth) and self.actor.load_core(pth):
                print(f"  transplanted movement CORE from {C.MOVE_CORE_PATH}")
            else:
                print(f"  WARNING: no movement core at {C.MOVE_CORE_PATH}; "
                      f"the branch will be a random frozen map")
        self.critic = PrefixCritic(self.actor.z_dim, num_actions).to(self.device)
        # ONE optimizer over everything. The encoder receives gradients from
        # both the policy loss and the critic/aux losses; two optimizers would
        # each keep their own Adam moments for those shared weights and fight.
        # Param groups keep the two learning rates.
        self.enc_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.crit_params = list(self.critic.parameters())
        self.opt = torch.optim.Adam([
            {"params": self.enc_params, "lr": C.ACTOR_LR},
            {"params": self.crit_params, "lr": C.CRITIC_LR},
        ])
        # CCA hindsight baseline: its OWN optimizer and its own encoder. Kept
        # off the shared trunk on purpose — the shared encoder feeds the
        # policy, and hindsight gradients there would leak future information
        # into the actor's features.
        self.hb = None
        if getattr(C, "HCA", False):
            self.hb = HindsightBaseline(state_dim, C.HIDDEN[-1],
                                        num_actions).to(self.device)
            self.hb_opt = torch.optim.Adam(self.hb.parameters(), lr=C.CRITIC_LR)
        self.kl_coef = C.KL_COEF
        self.updates = 0
        self.stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    def warm_start_from(self, other: "PPOAgent"):
        self.actor.load_state_dict(other.actor.state_dict())
        self.critic.load_state_dict(other.critic.state_dict())

    # ── acting ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act_batch(self, states: np.ndarray, h: torch.Tensor):
        """Sample actions AND advance the critic's prefix state.

        Returns (actions, logps, values, outcome_probs, new_h). The critic runs online here
        because V(prefix_t) needs the hidden state as it actually was at
        collection time — the trainer carries `h` across cycles and the update
        replays from it.
        """
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        z = self.actor.features(x)                  # shared encoding, computed once
        logits = self.actor.out(z)
        logp_all = F.log_softmax(logits, dim=1)
        a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
        logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        # eval(): dropout OFF during collection, so the V that goes into the
        # advantages is deterministic and reproducible by the replay.
        was = self.critic.training
        self.critic.eval()
        clog, v, h = self.critic.step(z, h)
        self.critic.train(was)
        # The predicted DISTRIBUTION rides along: it is the bootstrap target
        # for rows whose episode has not resolved by the end of the window.
        probs = torch.softmax(clog, dim=-1)
        return (a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy(),
                probs.cpu().numpy(), h)

    @torch.no_grad()
    def act_actions(self, net, states: np.ndarray, greedy: bool = False):
        """Actions only, from an arbitrary frozen actor (pool opponents). Runs
        on the NET's device — pool nets stay on CPU even when the agent trains
        on GPU (many small batches lose to dispatch overhead there)."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(
            next(net.parameters()).device)
        logits = net(x)
        if greedy:
            return logits.argmax(dim=1).cpu().numpy()
        return torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(1).cpu().numpy()

    def update_hindsight(self, batches):
        """Fit V(s_t, Phi_t) on the same padded episode batches, and return the
        per-row baseline scattered back into pool-row order.

        Trained on the SAME one-hot outcome target as the forward critic, so
        the two are directly comparable — the only difference is that this one
        may look at the future. The adversary shares the loss via gradient
        reversal, so one backward does both jobs."""
        if self.hb is None or not batches:
            return None, {}
        dev = self.device
        to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
        n_rows = 1 + max(int(b["row_index"].max()) for b in batches)
        vh = np.zeros(n_rows, dtype=np.float32)
        cl, al, ac = [], [], []
        n_ep = max(1, C.CRITIC_EPOCHS)
        for ep_i in range(n_ep):
            last = ep_i == n_ep - 1
            for bt in batches:
                obs = to(bt["states"])
                tgt = to(bt["targets"])
                m = to(bt["target_mask"], torch.bool)
                acts = to(bt["aux_self"], torch.long)   # a_t at this row
                if not bool(m.any()):
                    continue
                with torch.no_grad():
                    z = self.actor.features(obs)           # detached: policy features
                logits, adv_logits = self.hb(z, obs, m)
                loss = -(tgt[m] * F.log_softmax(logits[m], dim=-1)).sum(-1).mean()
                # Adversary tries to recover a_t; grad reversal makes Phi fight it.
                a_loss = F.cross_entropy(adv_logits[m], acts[m])
                self.hb_opt.zero_grad(set_to_none=True)
                (loss + a_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.hb.parameters(), C.MAX_GRAD_NORM)
                self.hb_opt.step()
                cl.append(loss.detach()); al.append(a_loss.detach())
                with torch.no_grad():
                    ac.append((logits[m].argmax(-1) == tgt[m].argmax(-1)).float().mean())
                    if last:
                        # Record V_h from the SAME forward pass rather than a
                        # separate sweep over every batch afterwards. The values
                        # are one gradient step stale, which is irrelevant for a
                        # baseline and saves a full pass per update.
                        v = self.hb.value_of(logits).cpu().numpy()
                        ri, mm = bt["row_index"], bt["target_mask"]
                        vh[ri[mm]] = v[mm]
        mean = lambda x: (torch.stack(x).mean().item() if x else 0.0)
        return vh, {"hca_loss": mean(cl), "hca_acc": mean(ac), "hadv": mean(al)}

    def update_critic_episodes(self, batches):
        """Critic pass over pooled WHOLE episodes.

        Every column of every batch is one complete episode, so h0 = 0 and no
        done-mask is needed inside a sequence — there is no boundary to mask.
        Padding past an episode's end is excluded from every loss by the batch
        mask. Long episodes still go through chunked TBPTT so activation memory
        follows CRITIC_BPTT_CHUNK rather than the episode length.
        """
        dev = self.device
        acc = {k: [] for k in ("c", "pos", "opp", "self", "acc")}
        if not batches:
            return {k: 0.0 for k in
                    ("critic_loss", "outcome_acc", "aux_loss", "aux_opp", "aux_self")}
        to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
        chunk = max(1, C.CRITIC_BPTT_CHUNK or 10 ** 9)
        for _ in range(C.CRITIC_EPOCHS):
            for bt in batches:
                obs_np, T, Bn = bt["states"], bt["states"].shape[0], bt["states"].shape[1]
                # Cap the SEQUENCE dimension. Episode buckets are wildly uneven
                # -- a short-episode bucket holds thousands of sequences -- and
                # unrolling a whole bucket at once was 3.3 GB of activations,
                # which is what blew up MPS. Memory is
                #   chunk x seqs x hidden x layers x ~10 tensors x 4B
                # so cap seqs from the budget and step through the bucket.
                spm = _critic_seqs_per_mb(Bn, min(T, chunk))
                # Pad T up to a whole number of chunks so every chunk tensor has
                # the SAME shape. torch-MPS compiles and caches a graph per
                # shape and never evicts, so a ragged final chunk on every
                # bucket would grow that cache without bound.
                Tp = -(-T // chunk) * chunk
                for s0 in range(0, Bn, spm):
                    g = slice(s0, min(Bn, s0 + spm))
                    nb_ = g.stop - g.start
                    def pad(x, fill=0):
                        y = np.full((Tp, nb_) + x.shape[2:], fill, dtype=x.dtype)
                        y[:T] = x[:, g]
                        return y
                    obs = to(pad(obs_np))
                    tgt = to(pad(bt["targets"]))
                    lab_m = to(pad(bt["target_mask"], False), torch.bool)
                    a_pos = to(pad(bt["aux_pos"]))
                    a_pos_m = to(pad(bt["aux_pos_mask"], False), torch.bool)
                    a_opp = to(pad(bt["aux_opp"]), torch.long)
                    a_opp_m = to(pad(bt["aux_opp_mask"], False), torch.bool)
                    a_slf = to(pad(bt["aux_self"]), torch.long)
                    a_slf_m = to(pad(bt["aux_self_mask"], False), torch.bool)
                    idx = torch.arange(nb_, device=dev)
                    zeros_done = torch.zeros(chunk, nb_, device=dev)
                    h_c = self.critic.initial_state(nb_, dev)
                    for cs in range(0, Tp, chunk):
                        sl = slice(cs, cs + chunk)
                        if not bool(lab_m[sl].any()):
                            # all-padding chunk: still advance the state, but
                            # skip the backward entirely
                            with torch.no_grad():
                                z = self.actor.features(obs[sl])
                                _l, _f, h_c = self.critic.unroll(z, h_c, zeros_done)
                            continue
                        if cs > 0:
                            h_c = h_c.detach()
                        z = self.actor.features(obs[sl])
                        logits, feat, h_c = self.critic.unroll(z, h_c, zeros_done)
                        self._critic_chunk_loss(logits, feat, idx, sl, tgt, lab_m,
                                                a_pos, a_pos_m, a_opp, a_opp_m,
                                                a_slf, a_slf_m, acc)
                    del obs, tgt, lab_m, a_pos, a_pos_m, a_opp, a_opp_m, a_slf, a_slf_m
        if dev.type == "mps":
            torch.mps.empty_cache()
        mean = lambda k: (torch.stack(acc[k]).mean().item() if acc[k] else 0.0)
        return {"critic_loss": mean("c"), "outcome_acc": mean("acc"),
                "aux_loss": mean("pos"), "aux_opp": mean("opp"),
                "aux_self": mean("self")}

    # ── critic update (sequences) ──────────────────────────────────────────
    def update_critic(self, roll: dict):
        """Cross-entropy on the episode outcome, plus the aux heads that share
        the GRU state. Minibatches by SEQUENCE (whole env columns), because a
        prefix cannot be split across minibatches without losing its history.
        """
        dev = self.device
        to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
        obs, done, h0 = to(roll["states"]), to(roll["dones"]), to(roll["h0"])
        # Target DISTRIBUTION per row: one-hot where the episode resolved,
        # the detached bootstrap distribution where it did not.
        tgt = to(roll["targets"])
        lab_m = to(roll["target_mask"], torch.bool)
        a_pos, a_pos_m = to(roll["aux_pos"]), to(roll["aux_pos_mask"], torch.bool)
        a_opp, a_opp_m = to(roll["aux_opp"], torch.long), to(roll["aux_opp_mask"], torch.bool)
        a_slf, a_slf_m = to(roll["aux_self"], torch.long), to(roll["aux_self_mask"], torch.bool)

        T, B = obs.shape[0], obs.shape[1]
        seqs_per_mb = _critic_seqs_per_mb(B, T)
        n_full = (B // seqs_per_mb) * seqs_per_mb
        acc = {k: [] for k in ("c", "pos", "opp", "self", "acc")}

        for _ in range(C.CRITIC_EPOCHS):
            order = torch.randperm(B, device=dev)
            for s in range(0, max(n_full, seqs_per_mb), seqs_per_mb):
                idx = order[s:s + seqs_per_mb]
                if idx.numel() < seqs_per_mb:
                    break
                chunk = min(T, C.CRITIC_BPTT_CHUNK) if C.CRITIC_BPTT_CHUNK else T
                h_c = h0[idx]
                for cs in range(0, T, chunk):
                    ce = min(T, cs + chunk)
                    sl = slice(cs, ce)
                    if cs > 0:
                        # carry across the chunk boundary, but DETACHED: this
                        # is what bounds activation memory to the chunk. The
                        # done-mask for the boundary row still applies.
                        h_c = (h_c * (1.0 - done[cs - 1, idx]).unsqueeze(-1)).detach()
                    z = self.actor.trunk(obs[sl, idx])
                    logits, feat, h_c = self.critic.unroll(z, h_c, done[sl, idx])
                    self._critic_chunk_loss(logits, feat, idx, sl, tgt, lab_m,
                                            a_pos, a_pos_m, a_opp, a_opp_m,
                                            a_slf, a_slf_m, acc)
        mean = lambda k: (torch.stack(acc[k]).mean().item() if acc[k] else 0.0)
        return {"critic_loss": mean("c"), "outcome_acc": mean("acc"),
                "aux_loss": mean("pos"), "aux_opp": mean("opp"),
                "aux_self": mean("self")}

    def _critic_chunk_loss(self, logits, feat, idx, sl, tgt, lab_m,
                           a_pos, a_pos_m, a_opp, a_opp_m, a_slf, a_slf_m, acc):
        """Loss + step for ONE BPTT chunk. Stepping per chunk (rather than
        accumulating over the window) is what keeps the graph short."""
        m = lab_m[sl, idx]
        if not m.any():
            return
        # Soft cross-entropy: -sum_c p_target(c) log p_pred(c). With a one-hot
        # target this is exactly the hard cross-entropy, so resolved and
        # bootstrapped rows need no branch.
        mt = tgt[sl, idx][m]
        loss = C.VALUE_COEF * (
            -(mt * F.log_softmax(logits[m], dim=-1)).sum(-1).mean())
        acc["c"].append(loss.detach())
        with torch.no_grad():
            acc["acc"].append((logits[m].argmax(-1) == mt.argmax(-1)).float().mean())

        pos_h, opp_h, self_h = self.critic.heads_from(feat)
        mp = a_pos_m[sl, idx]
        if C.AUX_NEXT_POS_COEF > 0 and mp.any():
            l = F.mse_loss(pos_h[mp], a_pos[sl, idx][mp])
            loss = loss + C.AUX_NEXT_POS_COEF * l
            acc["pos"].append(l.detach())
        mo = a_opp_m[sl, idx]
        if C.AUX_OPP_ACTION_COEF > 0 and mo.any():
            l = F.cross_entropy(opp_h[mo], a_opp[sl, idx][mo])
            loss = loss + C.AUX_OPP_ACTION_COEF * l
            acc["opp"].append(l.detach())
        ms = a_slf_m[sl, idx]
        if C.AUX_SELF_ACTION_COEF > 0 and ms.any():
            l = F.cross_entropy(self_h[ms], a_slf[sl, idx][ms])
            loss = loss + C.AUX_SELF_ACTION_COEF * l
            acc["self"].append(l.detach())

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.crit_params, C.MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(self.enc_params, C.MAX_GRAD_NORM)
        self.opt.step()

    # ── actor update (flat rows) ───────────────────────────────────────────
    def update_actor(self, rollout: dict, entropy_coef: float = C.ENTROPY_COEF_END):
        dev = self.device
        # Trim to a multiple of 2048 rows: torch-MPS caches a compiled graph per
        # tensor shape forever, so a different row count every update would leak
        # without bound. Only quantise above one bucket — trimming a small
        # rollout would round it to ZERO and silently skip the update.
        n = rollout["states"].shape[0]
        if n >= 2048:
            n -= n % 2048
        if n == 0:
            return dict(self.stats)
        states = torch.from_numpy(rollout["states"][:n]).to(dev)
        actions = torch.from_numpy(rollout["actions"][:n]).to(dev)
        old_logp = torch.from_numpy(rollout["logps"][:n]).to(dev)
        adv_np = rollout["advantages"][:n]
        if C.NORMALIZE_ADV:
            # Normalize in numpy: a variable-length reduction on MPS would be
            # one more per-shape graph in the never-evicted cache.
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv = torch.from_numpy(np.ascontiguousarray(adv_np)).to(dev)

        a_losses, ents, kls = [], [], []
        epochs_run, stopped_early = 0, False
        for _ in range(C.EPOCHS):
            perm = torch.randperm(n, device=dev)
            epoch_kls = []
            for s in range(0, n, C.MINIBATCH):
                idx = perm[s:s + C.MINIBATCH]
                logits = self.actor(states[idx])
                logp_all = F.log_softmax(logits, dim=1)
                logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                logratio = logp - old_logp[idx]
                ratio = logratio.exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - C.CLIP_EPS, 1 + C.CLIP_EPS) * adv[idx]
                entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()
                # k3 estimator: unbiased, far lower variance than -logratio,
                # and never negative (which -logratio can be).
                approx_kl = ((ratio - 1) - logratio).mean()
                loss = -torch.min(s1, s2).mean() - entropy_coef * entropy
                if self.kl_coef > 0.0:
                    loss = loss + self.kl_coef * approx_kl

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.enc_params, C.MAX_GRAD_NORM)
                self.opt.step()
                # Keep stats on-device; float() here would force a host sync
                # that stalls the MPS pipeline every minibatch (pure logging).
                a_losses.append(loss.detach())
                ents.append(entropy.detach())
                epoch_kls.append(approx_kl.detach())

            epochs_run += 1
            kls.extend(epoch_kls)
            # Early stop, checked once per EPOCH (one host sync per epoch, not
            # per minibatch). Fires only when this update already stepped past
            # TARGET_KL, so the remaining epochs would compound an overshoot.
            if C.TARGET_KL is not None and epoch_kls:
                if torch.stack(epoch_kls).mean().item() > C.TARGET_KL:
                    stopped_early = True
                    break

        if not a_losses:
            return dict(self.stats)
        mean_kl = torch.stack(kls).mean().item() if kls else 0.0
        # Schulman's 1.5x rule: chase KL_TARGET from whichever side we missed.
        if self.kl_coef > 0.0 and C.KL_ADAPTIVE:
            if mean_kl > 1.5 * C.KL_TARGET:
                self.kl_coef = min(C.KL_COEF_MAX, self.kl_coef * 2.0)
            elif mean_kl < C.KL_TARGET / 1.5:
                self.kl_coef = max(C.KL_COEF_MIN, self.kl_coef / 2.0)
        return {"actor_loss": torch.stack(a_losses).mean().item(),
                "entropy": torch.stack(ents).mean().item(),
                "ent_coef": entropy_coef, "kl": mean_kl,
                "kl_coef": self.kl_coef, "epochs": epochs_run,
                "kl_stopped": stopped_early}

    def update(self, actor_roll: dict, critic_roll: dict,
               entropy_coef: float = C.ENTROPY_COEF_END):
        stats = self.update_critic(critic_roll)
        stats.update(self.update_actor(actor_roll, entropy_coef))
        self.updates += 1
        self.stats = stats
        return stats

    # ── save / load (actor records feed play3.mjs) ─────────────────────────
    def serialize(self) -> dict:
        return {
            "actor": self.actor.to_records(),
            "critic": self.critic.to_records(),
            "updates": self.updates,
            "klCoef": self.kl_coef,
        }

    def load_state(self, obj: dict):
        self.actor.load_records(obj["actor"])
        crit = obj.get("critic")
        if crit is not None:
            if not isinstance(crit, dict):
                raise ValueError(
                    "this checkpoint's critic is a tfjs MLP record list, i.e. a "
                    "bonk2/ppo2/ppo3 checkpoint. ppo4's critic is a prefix GRU "
                    "and is not convertible — start a fresh run.")
            self.critic.load_records(crit)
        self.updates = obj.get("updates", 0)
        self.kl_coef = obj.get("klCoef", C.KL_COEF)
