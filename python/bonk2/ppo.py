"""bonk2 PPO: clipped surrogate + GAE, categorical over the 18 joint actions,
separate actor/critic MLPs with LayerNorm (tfjs-compatible records, so actors
deploy straight into play3.mjs). All hyperparameters live in bonk2.config.
"""

import numpy as np
import torch
import torch.nn.functional as F

from bonk.networks import MLP

from . import config as C


def entropy_coef_at(episode: int) -> float:
    """Anneal START -> END over ENTROPY_DECAY_EPISODES, then hold at END.
    "linear": constant absolute decrease per episode.
    "exponential": constant RATIO per episode (geometric) — drops fast early,
    long low tail. Reaches END exactly at ENTROPY_DECAY_EPISODES either way.
    Exponential requires START, END > 0."""
    frac = min(1.0, episode / C.ENTROPY_DECAY_EPISODES)
    if getattr(C, "ENTROPY_DECAY_MODE", "linear") == "exponential":
        return C.ENTROPY_COEF_START * (C.ENTROPY_COEF_END / C.ENTROPY_COEF_START) ** frac
    return C.ENTROPY_COEF_START + (C.ENTROPY_COEF_END - C.ENTROPY_COEF_START) * frac


def compute_gae(rewards, values, dones, last_value):
    """GAE(gamma, lambda) over one stream; last_value bootstraps a cut-off
    (non-terminal) tail."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    ret = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        next_v = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + C.GAMMA * next_v * nonterminal - values[t]
        gae = delta + C.GAMMA * C.GAE_LAMBDA * nonterminal * gae
        adv[t] = gae
        ret[t] = gae + values[t]
    return adv, ret


class PPOAgent:
    def __init__(self, state_dim: int, num_actions: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.shared = C.SHARED_TRUNK
        self.n_horizons = len(C.AUX_POS_HORIZONS)
        # The actor is a plain MLP (trunk + policy head) in BOTH modes. That is
        # what the league freezes, what export_model ships and what play3.mjs
        # rebuilds, so keeping it a real MLP is what makes SHARED_TRUNK
        # invisible to all of them.
        self.actor = MLP(state_dim, C.HIDDEN, num_actions).to(self.device)
        h = C.HIDDEN[-1]

        def head(out_dim):
            lin = torch.nn.Linear(h, out_dim).to(self.device)
            torch.nn.init.normal_(lin.weight, 0.0, 0.01)
            torch.nn.init.zeros_(lin.bias)
            return lin

        if self.shared:
            # One trunk (the actor's) feeds policy, value and aux.
            self.critic = None
            self.v_head = head(1)
            value_params = list(self.v_head.parameters())
        else:
            self.critic = MLP(state_dim, C.HIDDEN, 1).to(self.device)
            self.v_head = None
            value_params = list(self.critic.parameters())
        # Aux stays OUTSIDE any MLP: to_records()/from_records() infer
        # architecture from kernel shapes, so an extra head inside would corrupt
        # that and break every checkpoint loader (play3.mjs included).
        self.aux_head = head(2 * self.n_horizons)

        # One optimizer, two param groups. In separate mode the groups are
        # disjoint and each loss touches only its own group, so a single
        # combined backward gives gradients identical to the old two-optimizer
        # version. In shared mode the trunk lives in the actor group, so it is
        # stepped at ACTOR_LR while the value/aux heads keep CRITIC_LR.
        self.actor_params = list(self.actor.parameters())
        self.value_params = value_params + list(self.aux_head.parameters())
        self.opt = torch.optim.Adam([
            {"params": self.actor_params, "lr": C.ACTOR_LR},
            {"params": self.value_params, "lr": C.CRITIC_LR},
        ])
        self.kl_coef = C.KL_COEF
        self.updates = 0
        self.stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    def _forward_all(self, x):
        """(logits, value, trunk features for the aux heads) in one place, so
        the update loop has a single code path across both trunk modes."""
        if self.shared:
            f = self.actor.trunk(x)
            return self.actor.out(f), self.v_head(f).squeeze(-1), f
        f = self.critic.trunk(x)
        return self.actor(x), self.critic.out(f).squeeze(-1), f

    def warm_start_from(self, other: "PPOAgent"):
        """Copy every learnable tensor (exploiter warm start)."""
        self.actor.load_state_dict(other.actor.state_dict())
        self.aux_head.load_state_dict(other.aux_head.state_dict())
        if self.shared:
            self.v_head.load_state_dict(other.v_head.state_dict())
        else:
            self.critic.load_state_dict(other.critic.state_dict())

    # ── acting ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act_batch(self, states: np.ndarray):
        """Sample actions for the learner: returns (actions, logps, values)."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        # Shared mode gets the value for free here — one trunk pass instead of
        # the two a separate critic needs.
        logits, v, _ = self._forward_all(x)
        logp_all = F.log_softmax(logits, dim=1)
        a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
        logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    @torch.no_grad()
    def act_actions(self, net, states: np.ndarray, greedy: bool = False):
        """Actions only, from an arbitrary frozen actor (pool opponents).
        Runs on the NET's device — pool nets stay on CPU even when the agent
        trains on GPU (many small batches lose to dispatch overhead there)."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(
            next(net.parameters()).device)
        logits = net(x)
        if greedy:
            return logits.argmax(dim=1).cpu().numpy()
        return torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(1).cpu().numpy()

    # ── PPO update ─────────────────────────────────────────────────────────────
    def update(self, rollout: dict, entropy_coef: float = C.ENTROPY_COEF_END):
        """rollout: numpy arrays states [N,D], actions [N], logps [N],
        values [N], advantages [N], returns [N]."""
        dev = self.device
        # Trim to a multiple of 2048 rows: torch-MPS caches a compiled graph
        # per tensor shape forever, so a different row count every update
        # (variable opponent-stream size) leaks memory without bound. Trimming
        # bounds the shape set; the dropped tail is <2% of random rows.
        n = rollout["states"].shape[0]
        # Only quantise when there is more than one bucket's worth: trimming a
        # small rollout to a multiple of 2048 would round it to ZERO, silently
        # skipping the update. Shape churn only costs memory at production
        # scale, where n is always far above this.
        if n >= 2048:
            n -= n % 2048
        states = torch.from_numpy(rollout["states"][:n]).to(dev)
        actions = torch.from_numpy(rollout["actions"][:n]).to(dev)
        old_logp = torch.from_numpy(rollout["logps"][:n]).to(dev)
        old_v = torch.from_numpy(rollout["values"][:n]).to(dev)
        returns = torch.from_numpy(rollout["returns"][:n]).to(dev)
        adv_np = rollout["advantages"][:n]
        if C.NORMALIZE_ADV:
            # Normalize in numpy: a variable-length reduction on MPS would be
            # one more per-shape graph in the never-evicted cache.
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv = torch.from_numpy(np.ascontiguousarray(adv_np)).to(dev)
        # Auxiliary next-position target (metres) + validity mask. Absent when
        # disabled, or when the trainer had no next-observation to diff against.
        aux_t = aux_m = None
        if C.AUX_NEXT_POS_COEF > 0.0 and rollout.get("aux_target") is not None:
            aux_t = torch.from_numpy(rollout["aux_target"][:n]).to(dev)
            aux_m = torch.from_numpy(rollout["aux_mask"][:n]).to(dev)
        a_losses, c_losses, ents, aux_losses, kls = [], [], [], [], []
        epochs_run, stopped_early = 0, False
        for _ in range(C.EPOCHS):
            perm = torch.randperm(n, device=dev)
            epoch_kls = []
            for s in range(0, n, C.MINIBATCH):
                idx = perm[s:s + C.MINIBATCH]
                mb_s = states[idx]

                # One trunk pass feeds policy, value and aux in shared mode;
                # in separate mode this is the critic trunk and the actor runs
                # its own. Either way `feat` is what the aux heads read.
                logits, v, feat = self._forward_all(mb_s)
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
                actor_loss = -torch.min(s1, s2).mean() - entropy_coef * entropy
                if self.kl_coef > 0.0:
                    actor_loss = actor_loss + self.kl_coef * approx_kl

                if C.CLIP_VALUE_LOSS:
                    v_clip = old_v[idx] + (v - old_v[idx]).clamp(-C.CLIP_EPS,
                                                                 C.CLIP_EPS)
                    critic_loss = torch.max((v - returns[idx]) ** 2,
                                            (v_clip - returns[idx]) ** 2).mean()
                else:
                    critic_loss = F.mse_loss(v, returns[idx])
                critic_loss = C.VALUE_COEF * critic_loss

                if aux_t is not None:
                    m = aux_m[idx]                       # [mb, H]
                    if m.any():
                        # [mb, H, 2]; the boolean [mb, H] mask selects whole
                        # (dx, dy) pairs, so a horizon that ran past an episode
                        # boundary contributes nothing while the horizons that
                        # fit still train on that row.
                        pred = self.aux_head(feat).view(-1, self.n_horizons, 2)
                        aux_loss = F.mse_loss(pred[m], aux_t[idx][m])
                        critic_loss = critic_loss + C.AUX_NEXT_POS_COEF * aux_loss
                        aux_losses.append(aux_loss.detach())

                # ONE backward over the combined objective. With separate nets
                # the two param groups are disjoint and each term touches only
                # its own, so this reproduces the previous two-optimizer version
                # bit for bit (verified); with a shared trunk it is the whole
                # point — value and aux gradients reach the policy's features.
                # Clipping stays PER GROUP so MAX_GRAD_NORM keeps meaning what
                # it did before.
                #
                # ONE deliberate change: aux_head is now inside the clipped
                # group. It used to sit in critic_opt but was left out of
                # clip_grad_norm_, so it was the only trainable tensor taking
                # unclipped steps — that was an oversight, not a design.
                self.opt.zero_grad(set_to_none=True)
                (actor_loss + critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_params, C.MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(self.value_params, C.MAX_GRAD_NORM)
                self.opt.step()

                # Keep stats on-device; float() here would force a host sync
                # that stalls the MPS pipeline every minibatch (pure logging).
                a_losses.append(actor_loss.detach())
                c_losses.append(critic_loss.detach())
                ents.append(entropy.detach())
                epoch_kls.append(approx_kl.detach())

            epochs_run += 1
            kls.extend(epoch_kls)
            # Early stop, checked once per epoch (one host sync per epoch, not
            # per minibatch). Fires only when this update already stepped
            # further than TARGET_KL allows, so the remaining epochs would be
            # compounding an overshoot.
            if C.TARGET_KL is not None and epoch_kls:
                if torch.stack(epoch_kls).mean().item() > C.TARGET_KL:
                    stopped_early = True
                    break

        if not a_losses:      # nothing trained (degenerate rollout) — keep last
            return self.stats
        self.updates += 1
        mean_kl = torch.stack(kls).mean().item() if kls else 0.0
        # Schulman's 1.5x rule: chase KL_TARGET from whichever side we missed.
        if self.kl_coef > 0.0 and C.KL_ADAPTIVE:
            if mean_kl > 1.5 * C.KL_TARGET:
                self.kl_coef = min(C.KL_COEF_MAX, self.kl_coef * 2.0)
            elif mean_kl < C.KL_TARGET / 1.5:
                self.kl_coef = max(C.KL_COEF_MIN, self.kl_coef / 2.0)
        # One sync per metric at the very end, instead of 3 per minibatch.
        self.stats = {
            "actor_loss": torch.stack(a_losses).mean().item(),
            "critic_loss": torch.stack(c_losses).mean().item(),
            "entropy": torch.stack(ents).mean().item(),
            "ent_coef": entropy_coef,
            "aux_loss": (torch.stack(aux_losses).mean().item()
                         if aux_losses else 0.0),
            "kl": mean_kl,
            "kl_coef": self.kl_coef,
            "epochs": epochs_run,
            "kl_stopped": stopped_early,
        }
        return self.stats

    # ── save / load (actor records feed play3.mjs) ─────────────────────────────
    @staticmethod
    def _head_records(lin) -> dict:
        return {"w": lin.weight.detach().cpu().numpy().tolist(),
                "b": lin.bias.detach().cpu().numpy().tolist()}

    def _load_head(self, lin, rec, name: str) -> None:
        """Training-only heads: reinitialise rather than refuse on a width
        change, so a checkpoint from a different AUX_POS_HORIZONS still resumes.
        None of these ship — export_model takes the actor alone."""
        if not rec:
            return
        w = torch.tensor(rec["w"])
        if w.shape != lin.weight.shape:
            print(f"note: checkpoint {name} is {tuple(w.shape)}, this build "
                  f"wants {tuple(lin.weight.shape)} — reinitialising it")
            return
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(torch.tensor(rec["b"]))

    def serialize(self) -> dict:
        obj = {
            # Unchanged in both trunk modes, which is what keeps export_model,
            # play3.mjs and the league's frozen opponents working either way.
            "actor": self.actor.to_records(),
            "updates": self.updates,
            "sharedTrunk": self.shared,
            "klCoef": self.kl_coef,
            # Plain tensors, not tfjs records: training-only heads must never
            # look like part of an MLP's layer stack.
            "auxHead": self._head_records(self.aux_head),
        }
        if self.shared:
            obj["vHead"] = self._head_records(self.v_head)
        else:
            obj["critic"] = self.critic.to_records()
        return obj

    def load_state(self, obj: dict):
        # Default False: checkpoints written before this flag existed always
        # had separate nets.
        was_shared = obj.get("sharedTrunk", False)
        if was_shared != self.shared:
            raise ValueError(
                f"trunk mismatch: checkpoint has "
                f"{'a shared' if was_shared else 'separate'} trunk, "
                f"config.SHARED_TRUNK={self.shared} wants "
                f"{'a shared' if self.shared else 'separate'} one. The value "
                f"head and trunk weights are not interchangeable — start a "
                f"fresh run or flip SHARED_TRUNK back.")
        self.actor.load_records(obj["actor"])
        if self.shared:
            self._load_head(self.v_head, obj.get("vHead"), "value head")
        else:
            self.critic.load_records(obj["critic"])
        self.updates = obj.get("updates", 0)
        self.kl_coef = obj.get("klCoef", C.KL_COEF)
        # absent in pre-aux checkpoints; width-checked by _load_head
        self._load_head(self.aux_head, obj.get("auxHead"), "aux head")
