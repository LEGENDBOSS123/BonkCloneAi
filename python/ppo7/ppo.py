"""PPOAgent: clipped-surrogate PPO with GAE over the joint (action, duration)
policy, plus a categorical outcome critic.

Two heads are packed into ONE actor output vector — `[0:NUM_ACTIONS]` are the
joint-action logits, the rest are the FiGAR duration logits. One net keeps the
records format and the browser deploy simple; the split happens wherever the
policy is read (`_split`). The heads are conditionally independent given the
state, so their log-probs and entropies simply add.

The critic is a 3-way classifier over {win, draw, loss} rather than a scalar
regressor (config.CRITIC_ATOMS). That is only sound because GAMMA = 1 and the
reward is terminal-only, which makes the distributional Bellman backup a plain
backward propagation of the realised outcome.

There are TWO update paths, picked by `config.RECURRENT`:
  * `update`            flat rows, feedforward net (legacy TCN encoder).
  * `update_recurrent`  truncated BPTT over [T, E] sequences (minGRU encoder).
Both share the loss shape; they differ in how a minibatch is assembled (rows vs
whole env trajectories) and in which auxiliary heads run.

Everything tunable lives in ppo7.config; the entropy SCHEDULES live in
ppo7.schedules (re-exported here for callers that already import them from this
module). Actor records are tfjs-compatible, so a trained actor deploys straight
into the browser script.
"""

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .hca import HindsightBuffer, HindsightHead, train_hindsight
from .mingru import MinGRUNet
from .networks import TemporalNet
from .schedules import (duration_entropy_coef_at,  # noqa: F401  (re-export)
                        entropy_coef_at)


class PPOAgent:
    def __init__(self, state_dim: int, num_actions: int, device: str = "cpu",
                 atoms=None, is_exploiter: bool = False, is_staller: bool = False):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        # Exploiters get a LOOSER trust region + bigger steps: they best-respond
        # to a FROZEN main and must DIVERGE from it to find a hole, so a
        # conservative clip/LR just keeps them hugging the main. Aggressive
        # learners (AlphaStar-style). The main stays conservative.
        self.is_exploiter = is_exploiter
        self.clip_eps = (C.EXPLOITER_CLIP_EPS if is_exploiter else C.CLIP_EPS)
        self._actor_lr = (C.EXPLOITER_ACTOR_LR if is_exploiter else C.ACTOR_LR)
        self.epochs = (C.EXPLOITER_EPOCHS if is_exploiter else C.EPOCHS)
        # Parameter-space noise (exploiter only): a perturbed copy of the actor is
        # the BEHAVIOR policy for collection; the clean actor is what we train.
        # (behavior_actor is built later, once self.actor exists.)
        self.param_noise = is_exploiter and bool(getattr(C, "EXPLOITER_PARAM_NOISE", False))
        # Two heads packed into one output vector: [0:num_actions] = joint action
        # logits, [num_actions:] = duration logits. One MLP keeps records/deploy
        # simple; the split happens wherever the policy is read.
        self.n_dur = C.NUM_DURATIONS
        self.recurrent = bool(getattr(C, "RECURRENT", False))
        _Net = MinGRUNet if self.recurrent else TemporalNet
        _kw = {"gru_hidden": C.MINGRU_HIDDEN} if self.recurrent else {}
        self.actor = _Net(state_dim, C.HIDDEN,
                          num_actions + self.n_dur, **_kw).to(self.device)
        # `atoms` lets an EXPLOITER use its own outcome values (a draw is worth
        # 0 to it, not -1) while sharing all other machinery. Must be set
        # BEFORE the critic, which is sized from it.
        self.atom_vals = tuple(atoms if atoms is not None else C.CRITIC_ATOMS)
        self.atoms = torch.tensor(self.atom_vals, dtype=torch.float32,
                                  device=self.device)
        # One logit per atom when categorical, else a scalar value.
        self.cat = bool(C.CRITIC_CATEGORICAL)
        self.critic = _Net(state_dim, C.HIDDEN,
                           len(self.atom_vals) if self.cat else 1, **_kw).to(self.device)
        # ── position aux head (feature #1 under test) ──────────────────────
        # Predicts the agent's OWN displacement at several horizons from the
        # CRITIC's trunk. It hangs off the critic, never the actor, so it
        # cannot perturb the policy directly -- only sharpen V, which reaches
        # the policy second-hand through better advantages. Self-supervised
        # from data the game already produced; no human picks the target, so
        # it stays inside the reward constraint.
        # Time-to-resolution head (Lc0 MLH analogue), on the CRITIC trunk.
        self.time_head = torch.nn.Linear(C.HIDDEN[-1], 1).to(self.device)
        torch.nn.init.zeros_(self.time_head.bias)
        torch.nn.init.normal_(self.time_head.weight, 0.0, 0.01)
        self.n_horizons = len(C.AUX_POS_HORIZONS)
        self.aux_head = torch.nn.Linear(C.HIDDEN[-1],
                                        2 * self.n_horizons).to(self.device)
        torch.nn.init.normal_(self.aux_head.weight, 0.0, 0.01)
        torch.nn.init.zeros_(self.aux_head.bias)
        # RUDDER's return-predictor g(s) -- same "ride on the critic trunk"
        # pattern as aux_head/time_head above (config.RUDDER_COEF/LOSS_COEF).
        self.rudder_head = torch.nn.Linear(C.HIDDEN[-1], 1).to(self.device)
        torch.nn.init.zeros_(self.rudder_head.bias)
        torch.nn.init.normal_(self.rudder_head.weight, 0.0, 0.01)
        # Multi-gamma auxiliary heads (config.AUX_GAMMA_VALUES/AUX_GAMMA_COEF)
        # -- same trunk-riding pattern, one extra scalar output per gamma.
        self.n_gammas = len(C.AUX_GAMMA_VALUES)
        self.gamma_head = torch.nn.Linear(C.HIDDEN[-1], self.n_gammas).to(self.device)
        torch.nn.init.zeros_(self.gamma_head.bias)
        torch.nn.init.normal_(self.gamma_head.weight, 0.0, 0.01)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
        # aux_head/time_head/rudder_head/gamma_head ride with the critic: same
        # trunk, same learning rate, and crucially NOT in the actor's optimizer.
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.aux_head.parameters())
            + list(self.time_head.parameters()) + list(self.rudder_head.parameters())
            + list(self.gamma_head.parameters()),
            lr=C.CRITIC_LR)
        if self.param_noise:
            import copy as _copy
            self.behavior_actor = _copy.deepcopy(self.actor)   # actor exists now
            for p in self.behavior_actor.parameters():
                p.requires_grad_(False)
            self.pn_sigma = C.PARAM_NOISE_SIGMA_INIT
        # ── adaptive entropy controller (MAIN head only) ────────────────────
        # Holds H at ENTROPY_TARGET_H by nudging the coef after each update.
        # Exploiters keep their scheduled two-stage coef (must commit low), so
        # the controller is disabled for them.
        self.ent_adaptive = bool(getattr(C, "ENTROPY_ADAPTIVE", False)) and not is_exploiter
        self.ent_coef = float(getattr(C, "ENTROPY_COEF_INIT", C.ENTROPY_COEF_START))
        # Hold-bonus weights: w_d = log2(hold) = duration index (DURATIONS are
        # powers of 2), and the hold length in cycles (for the decisions/game stat).
        self._dur_w = torch.arange(self.n_dur, dtype=torch.float32, device=self.device)
        self._dur_cycles = torch.tensor(C.DURATIONS, dtype=torch.float32, device=self.device)
        # ── HCA (hca.py), EXPLOITERS ONLY (main agent excluded) ─────────────
        # See ppo7-hca-for-exploiters (memory): the documented main-agent
        # failure (HCA sharpens onto stalling before the policy has learned to
        # engage) is not a failure mode for a STALLER -- "converge onto
        # loss-avoidance" IS its objective, not a degenerate one. For a
        # COMBAT exploiter it's a live, untested risk (draw is still valued at
        # -1 same as a loss in EXPLOITER_ATOMS, so the same stall-toward-safety
        # sharpening COULD recur) -- on here anyway per explicit instruction;
        # watch entropy/hdiv/gate-hit rate for combat exploiters specifically.
        self.is_staller = is_staller
        self.hca_head = self.hca_opt = self.hca_buf = self.hca_rng = None
        if is_exploiter and bool(getattr(C, "HCA", False)):
            self.hca_head = HindsightHead(state_dim, num_actions).to(self.device)
            self.hca_opt = torch.optim.Adam(self.hca_head.parameters(), lr=C.HCA_LR)
            self.hca_buf = HindsightBuffer(C.HCA_BUFFER, state_dim)
            self.hca_rng = np.random.default_rng()
        self.updates = 0
        self.stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    def add_hca_rows(self, states, actions, z, valid) -> None:
        """Feed FINISHED-episode (s, a, z) rows into the HCA buffer. `valid`
        selects which rows actually carry a resolved label (see
        `targets.hca_episode_labels`); a no-op when HCA isn't active for this
        agent."""
        if self.hca_buf is None or not valid.any():
            return
        self.hca_buf.add(states[valid], actions[valid], z[valid])

    def train_hca(self) -> dict:
        """One classifier-training step from the buffer; {} if inert (buffer
        below HCA_MIN_ROWS) or HCA isn't active for this agent."""
        if self.hca_head is None:
            return {}
        return train_hindsight(self.hca_head, self.hca_opt, self.hca_buf,
                               self.device, self.hca_rng)

    @torch.no_grad()
    def adapt_entropy(self, measured_h: float):
        """Nudge the entropy coef toward holding H at ENTROPY_TARGET_H: raise it
        when H has dropped below target (under-exploring / collapsing), lower it
        when H is above (over-exploring).

        `measured_h` is the COMBINED action+duration entropy the update reports,
        so the hold bonus (which peaks the duration head) mildly self-dampens
        through this controller. Documented coupling, not a bug; the clean fix
        is to feed it the action entropy alone.
        """
        if not self.ent_adaptive:
            return self.ent_coef
        if measured_h < C.ENTROPY_TARGET_H:
            self.ent_coef *= C.ENTROPY_ADAPT_RATE
        else:
            self.ent_coef /= C.ENTROPY_ADAPT_RATE
        self.ent_coef = float(min(C.ENTROPY_COEF_MAX,
                                  max(C.ENTROPY_COEF_MIN, self.ent_coef)))
        return self.ent_coef

    # ── acting ─────────────────────────────────────────────────────────────────
    def _split(self, logits):
        """(action_logits, duration_logits) from the packed actor output."""
        return logits[:, :self.num_actions], logits[:, self.num_actions:]

    @torch.no_grad()
    def act_batch(self, states: np.ndarray):
        """Sample for the learner: (actions, durations, logps, values). The
        joint log-prob is logp(action) + logp(duration) -- the two heads are
        independent given the state."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        # Collect with the PERTURBED actor (the behavior policy) when param-noise
        # is on; its log-probs become old_logp, so the PPO ratio stays correct.
        act_net = self.behavior_actor if self.param_noise else self.actor
        la, ld = self._split(act_net(x))
        lpa, lpd = F.log_softmax(la, 1), F.log_softmax(ld, 1)
        a = torch.multinomial(lpa.exp(), 1).squeeze(1)
        d = torch.multinomial(lpd.exp(), 1).squeeze(1)
        logp = (lpa.gather(1, a.unsqueeze(1)) + lpd.gather(1, d.unsqueeze(1))).squeeze(1)
        v = self.value_of(self.critic(x))
        return (a.cpu().numpy(), d.cpu().numpy(),
                logp.cpu().numpy(), v.cpu().numpy())

    @torch.no_grad()
    def act_step(self, states, ha, hc):
        """Recurrent collection step. states[n,D], ha/hc[n,H] previous hidden ->
        (actions, durations, logps, values, ha_new, hc_new) as numpy. The actor
        and critic each carry their OWN minGRU hidden state across decisions."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        hat = torch.from_numpy(np.ascontiguousarray(ha)).to(self.device)
        hct = torch.from_numpy(np.ascontiguousarray(hc)).to(self.device)
        logits, ha2 = self.actor.step(x, hat)
        la, ld = self._split(logits)
        lpa, lpd = F.log_softmax(la, 1), F.log_softmax(ld, 1)
        a = torch.multinomial(lpa.exp(), 1).squeeze(1)
        d = torch.multinomial(lpd.exp(), 1).squeeze(1)
        logp = (lpa.gather(1, a.unsqueeze(1)) + lpd.gather(1, d.unsqueeze(1))).squeeze(1)
        c_out, hc2 = self.critic.step(x, hct)
        v = self.value_of(c_out)
        return (a.cpu().numpy(), d.cpu().numpy(), logp.cpu().numpy(),
                v.cpu().numpy(), ha2.cpu().numpy(), hc2.cpu().numpy())

    @torch.no_grad()
    def act_actions_step(self, net, states, h):
        """A frozen/opponent minGRU net acts with carried hidden state (sampled).
        Returns (actions, durations, h_new) as numpy. Uses the NET's device (pool
        nets live on CPU)."""
        dev = next(net.parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(states)).to(dev)
        ht = torch.from_numpy(np.ascontiguousarray(h)).to(dev)
        logits, h2 = net.step(x, ht)
        la, ld = self._split(logits)
        a = torch.multinomial(F.softmax(la, 1), 1).squeeze(1)
        d = torch.multinomial(F.softmax(ld, 1), 1).squeeze(1)
        return a.cpu().numpy(), d.cpu().numpy(), h2.cpu().numpy()

    # ── parameter-space noise (exploiter) ───────────────────────────────────────
    @torch.no_grad()
    def resample_param_noise(self):
        """Refresh the behavior actor = clean actor + N(0, sigma) on every weight.
        Call once per rollout so exploration is coherent within the rollout."""
        if not self.param_noise:
            return
        self.behavior_actor.load_state_dict(self.actor.state_dict())
        for p in self.behavior_actor.parameters():
            p.add_(torch.randn_like(p) * self.pn_sigma)

    @torch.no_grad()
    def adapt_param_noise(self, states: np.ndarray):
        """Adjust sigma so the perturbed policy sits ~PARAM_NOISE_TARGET_KL from the
        clean one in action space (too-small = no exploration, too-big = random)."""
        if not self.param_noise or states.shape[0] == 0:
            return
        x = torch.from_numpy(np.ascontiguousarray(states[:4096])).to(self.device)
        la_c, _ = self._split(self.actor(x))
        la_p, _ = self._split(self.behavior_actor(x))
        lpc, lpp = F.log_softmax(la_c, 1), F.log_softmax(la_p, 1)
        kl = (lpc.exp() * (lpc - lpp)).sum(1).mean().item()
        if kl < C.PARAM_NOISE_TARGET_KL:
            self.pn_sigma *= C.PARAM_NOISE_ADAPT
        else:
            self.pn_sigma /= C.PARAM_NOISE_ADAPT
        return kl

    @torch.no_grad()
    def act_actions(self, net, states: np.ndarray, greedy: bool = False):
        """(actions, durations) from an arbitrary frozen actor (pool opponents).
        Runs on the NET's device — pool nets stay on CPU even when the agent
        trains on GPU (many small batches lose to dispatch overhead there)."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(
            next(net.parameters()).device)
        logits = net(x)
        la, ld = logits[:, :self.num_actions], logits[:, self.num_actions:]
        if greedy:
            return la.argmax(1).cpu().numpy(), ld.argmax(1).cpu().numpy()
        a = torch.multinomial(F.softmax(la, 1), 1).squeeze(1)
        d = torch.multinomial(F.softmax(ld, 1), 1).squeeze(1)
        return a.cpu().numpy(), d.cpu().numpy()

    # ── PPO update ─────────────────────────────────────────────────────────────
    def value_of(self, out):
        """critic output -> scalar V. Categorical: a convex combination of the
        atoms, hence bounded by construction."""
        if not self.cat:
            return out.squeeze(-1)
        return torch.softmax(out, dim=-1) @ self.atoms

    def update(self, rollout: dict, entropy_coef: float = C.ENTROPY_COEF_END,
               shaping_w: float = 0.0,
               duration_entropy_coef: float = C.DURATION_ENTROPY_COEF_END):
        """rollout: numpy arrays states [N,D], actions [N], logps [N],
        values [N], advantages [N], returns [N]."""
        dev = self.device
        # Trim to a multiple of 2048 rows: torch-MPS caches a compiled graph
        # per tensor shape forever, so a different row count every update
        # (variable opponent-stream size) leaks memory without bound. Trimming
        # bounds the shape set; the dropped tail is <2% of random rows.
        n = rollout["states"].shape[0]
        n -= n % 2048
        states = torch.from_numpy(rollout["states"][:n]).to(dev)
        actions = torch.from_numpy(rollout["actions"][:n]).to(dev)
        durations = torch.from_numpy(rollout["durations"][:n]).to(dev)
        # Actor trains only on FREE decision rows (held rows were forced repeats,
        # not policy samples -- their ratio would be off-policy garbage).
        free = torch.from_numpy(rollout["free"][:n].astype(np.bool_)).to(dev)
        old_logp = torch.from_numpy(rollout["logps"][:n]).to(dev)
        old_v = torch.from_numpy(rollout["values"][:n]).to(dev)
        returns = torch.from_numpy(rollout["returns"][:n]).to(dev)
        adv_np = rollout["advantages"][:n]
        if C.NORMALIZE_ADV:
            # Normalize in numpy: a variable-length reduction on MPS would be
            # one more per-shape graph in the never-evicted cache.
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        if shaping_w > 0.0 and rollout.get("ddec") is not None:
            # Reward APPROACHING (distance decreasing this step), not being
            # close. Current-closeness gave no gradient because a far agent
            # never samples close states (measured 0% contact); ddec rewards
            # every step toward the opponent, from any distance.
            ddec = rollout["ddec"][:n].astype(np.float32)
            ddec = (ddec - ddec.mean()) / (ddec.std() + 1e-8)
            adv_np = adv_np + shaping_w * ddec
        adv = torch.from_numpy(np.ascontiguousarray(adv_np)).to(dev)
        a_losses, c_losses, ents = [], [], []
        hold_lens = []
        aux_losses = []
        # Aux targets are optional: absent when the coefficient is 0 or the
        # trainer had no next-observation to difference against.
        oc_t = None
        if self.cat:
            if rollout.get("outcome_target") is None:
                raise ValueError("CRITIC_CATEGORICAL needs 'outcome_target' "
                                 "in the rollout")
            oc_t = torch.from_numpy(rollout["outcome_target"][:n]).to(dev)
        time_losses = []
        tt_t = None
        if C.CRITIC_TIME_COEF > 0.0 and rollout.get("time_target") is not None:
            tt_t = torch.from_numpy(rollout["time_target"][:n]).to(dev)
        aux_t = aux_m = None
        if C.AUX_NEXT_POS_COEF > 0.0 and rollout.get("aux_target") is not None:
            aux_t = torch.from_numpy(rollout["aux_target"][:n]).to(dev)
            aux_m = torch.from_numpy(rollout["aux_mask"][:n]).to(dev)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, C.MINIBATCH):
                idx = perm[s:s + C.MINIBATCH]
                mb_s = states[idx]

                la, ld = self._split(self.actor(mb_s))
                lpa, lpd = F.log_softmax(la, 1), F.log_softmax(ld, 1)
                # Joint log-prob of the (action, duration) pair; the two heads
                # are conditionally independent, so log-probs and entropies add.
                logp = (lpa.gather(1, actions[idx].unsqueeze(1))
                        + lpd.gather(1, durations[idx].unsqueeze(1))).squeeze(1)
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv[idx]
                # Per-head entropies: the two heads carry SEPARATE coefficients
                # so the duration head can be held open (it collapses under the
                # action head's smaller coef -- long holds go to 0% sampled).
                ent_a = -(lpa.exp() * lpa).sum(1)
                ent_d = -(lpd.exp() * lpd).sum(1)
                ent_row = ent_a + ent_d                      # combined, for the stat
                # Mask to FREE rows: held rows carry a forced (non-sampled)
                # action, so their surrogate/entropy must not touch the actor.
                fm = free[idx].float()
                denom = fm.sum().clamp(min=1.0)
                surr = -torch.min(s1, s2)
                ent_bonus = (entropy_coef * (ent_a * fm).sum()
                             + duration_entropy_coef * (ent_d * fm).sum())
                # Hold bonus: pull the duration head toward longer holds by a
                # STATE-INDEPENDENT per-duration weight (w_d = log2 hold). Maximized
                # (subtracted from the loss). The return gradient in `surr` still
                # pushes holds short where re-deciding matters, so this only wins
                # in states where holding is free -> learns WHEN to hold.
                hold_bonus = 0.0
                if C.HOLD_BONUS > 0.0:
                    hold_pref = (lpd.exp() * self._dur_w).sum(1)   # E[log2 hold] per row
                    hold_bonus = C.HOLD_BONUS * (hold_pref * fm).sum()
                actor_loss = ((surr * fm).sum() - ent_bonus - hold_bonus) / denom
                entropy = (ent_row * fm).sum() / denom
                # decisions/game proxy: mean sampled hold length (cycles) on FREE rows
                with torch.no_grad():
                    mh = (self._dur_cycles[durations[idx]] * fm).sum() / denom
                    hold_lens.append(float(mh))
                do_actor = (not C.FREEZE_ACTOR) and bool(free[idx].any())
                if do_actor:
                    self.actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                   C.MAX_GRAD_NORM)
                    self.actor_opt.step()

                cf = self.critic.trunk(mb_s)
                c_out = self.critic.out(cf)
                v = self.value_of(c_out)
                if C.CRITIC_TIME_COEF > 0.0 and tt_t is not None:
                    t_pred = self.time_head(cf).squeeze(-1)
                    t_loss = F.mse_loss(t_pred, tt_t[idx])
                    time_losses.append(float(t_loss.detach()))
                if self.cat:
                    # Cross-entropy against the outcome distribution. Soft
                    # targets, because a bootstrapped tail carries the detached
                    # next-state distribution rather than a one-hot.
                    critic_loss = -(oc_t[idx]
                                    * F.log_softmax(c_out, dim=-1)).sum(-1).mean()
                elif C.CLIP_VALUE_LOSS:
                    v_clip = old_v[idx] + (v - old_v[idx]).clamp(-C.CLIP_EPS,
                                                                 C.CLIP_EPS)
                    critic_loss = torch.max((v - returns[idx]) ** 2,
                                            (v_clip - returns[idx]) ** 2).mean()
                else:
                    critic_loss = F.mse_loss(v, returns[idx])
                critic_loss = C.VALUE_COEF * critic_loss
                if time_losses:
                    critic_loss = critic_loss + C.CRITIC_TIME_COEF * t_loss
                if aux_t is not None:
                    m = aux_m[idx]
                    if m.any():
                        # [mb, H, 2]; the [mb, H] bool mask selects whole
                        # (dx, dy) pairs, so a horizon that ran past an episode
                        # boundary contributes nothing while horizons that fit
                        # still train on that row.
                        pred = self.aux_head(cf).view(-1, self.n_horizons, 2)
                        aux_l = F.mse_loss(pred[m], aux_t[idx][m])
                        critic_loss = critic_loss + C.AUX_NEXT_POS_COEF * aux_l
                        aux_losses.append(float(aux_l.detach()))
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                               C.MAX_GRAD_NORM)
                self.critic_opt.step()

                a_losses.append(float(actor_loss.detach()))
                c_losses.append(float(critic_loss.detach()))
                ents.append(float(entropy.detach()))

        self.updates += 1
        self.stats = {
            "actor_loss": sum(a_losses) / len(a_losses),
            "critic_loss": sum(c_losses) / len(c_losses),
            "entropy": sum(ents) / len(ents),
            "ent_coef": entropy_coef,
            "mean_hold": (sum(hold_lens) / len(hold_lens)) if hold_lens else 1.0,
        }
        return self.stats

    def update_recurrent(self, buf, entropy_coef=C.ENTROPY_COEF_END,
                         duration_entropy_coef=C.DURATION_ENTROPY_COEF_END):
        """Truncated-BPTT PPO over the learner's [T,E] rollout. The minGRU hidden
        state is carried ACROSS rollouts during collection (reset only at episode
        ends), so each rollout is a contiguous chunk we re-forward from its stored
        start hidden `ha0`/`hc0`. Minibatches are over ENVIRONMENTS; each is a full
        length-T sequence. Loss is masked to rows >= BURN_IN (the leading steps
        only warm the recomputed hidden). Only the learner seat is trained."""
        dev = self.device
        to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
        T, E, D = buf["states"].shape
        S = to(buf["states"])                       # [T,E,D]
        A = to(buf["actions"], torch.long); Du = to(buf["durations"], torch.long)
        FR = to(buf["free"]);  OLP = to(buf["old_logp"]);  RET = to(buf["returns"])
        OC = to(buf["outcome_target"]) if self.cat else None
        RT = to(buf["rudder_target"]) if buf.get("rudder_target") is not None else None
        RV = to(buf["rudder_valid"]) if buf.get("rudder_valid") is not None else None
        GT = to(buf["gamma_target"]) if buf.get("gamma_target") is not None else None
        DON = to(buf["dones"])
        ha0 = to(buf["ha0"]); hc0 = to(buf["hc0"])  # [E,H]
        adv = buf["advantages"].astype(np.float32)
        if C.NORMALIZE_ADV:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ADV = to(adv)
        # env-major sequences [E,T,*]; reset zeros hidden at each new-episode step
        # (the row AFTER a done), matching the collection reset.
        St = S.transpose(0, 1).contiguous()          # [E,T,D]
        reset = torch.zeros(E, T, device=dev); reset[:, 1:] = DON.transpose(0, 1)[:, :-1]
        loss_mask = torch.zeros(T, device=dev); loss_mask[C.BURN_IN:] = 1.0   # [T]
        idx_env = np.arange(E); MB = max(1, C.MINIBATCH // max(1, T))
        a_losses, c_losses, ents, hold_lens, hca_hdivs = [], [], [], [], []
        ent_a_list, ent_d_list, rudder_losses, gamma_losses = [], [], [], []
        for _ in range(self.epochs):
            np.random.shuffle(idx_env)
            for s in range(0, E, MB):
                mb = idx_env[s:s + MB]
                xb = St[mb]                                          # [B,T,D]
                logits, _, _ = self.actor.forward_seq(xb, ha0[mb], reset[mb])
                c_out, _, c_trunk = self.critic.forward_seq(xb, hc0[mb], reset[mb])
                la, ld = logits[..., :self.num_actions], logits[..., self.num_actions:]
                lpa, lpd = F.log_softmax(la, -1), F.log_softmax(ld, -1)
                a_mb = A[:, mb].transpose(0, 1); d_mb = Du[:, mb].transpose(0, 1)
                logp = (lpa.gather(-1, a_mb.unsqueeze(-1))
                        + lpd.gather(-1, d_mb.unsqueeze(-1))).squeeze(-1)     # [B,T]
                old = OLP[:, mb].transpose(0, 1); advb = ADV[:, mb].transpose(0, 1)
                if self.hca_head is not None:
                    # Blend in HCA's per-action credit (exploiters only -- see
                    # the constructor). `logp_pi` is the CURRENT policy's
                    # action-only log-prob, recomputed fresh each minibatch --
                    # a deliberate simplification of the docstring's stated
                    # ideal (the exact behavior-policy log-prob at collection
                    # time), which would need a new stored field on every
                    # rollout row; HCA's own ratio clip bounds the resulting
                    # distortion the same way PPO's own ratio already
                    # tolerates a stale old_logp across epochs.
                    adv_hca, hdiv = self.hca_head.advantage(
                        xb.reshape(-1, xb.shape[-1]), a_mb.reshape(-1),
                        lpa.gather(-1, a_mb.unsqueeze(-1)).squeeze(-1).reshape(-1))
                    adv_hca = adv_hca.reshape(advb.shape)
                    advb = (1.0 - C.HCA_BLEND) * advb + C.HCA_BLEND * adv_hca
                    hca_hdivs.append(hdiv)
                ratio = (logp - old).exp()
                surr = -torch.min(ratio * advb,
                                  ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advb)
                ent_a = -(lpa.exp() * lpa).sum(-1); ent_d = -(lpd.exp() * lpd).sum(-1)
                # actor trains on FREE rows past burn-in only
                fm = FR[:, mb].transpose(0, 1) * loss_mask[None, :]
                denom = fm.sum().clamp(min=1.0)
                ent_bonus = (entropy_coef * (ent_a * fm).sum()
                             + duration_entropy_coef * (ent_d * fm).sum())
                hold_bonus = 0.0
                if C.HOLD_BONUS > 0.0:
                    hold_pref = (lpd.exp() * self._dur_w).sum(-1)
                    hold_bonus = C.HOLD_BONUS * (hold_pref * fm).sum()
                actor_loss = ((surr * fm).sum() - ent_bonus - hold_bonus) / denom
                # critic: every row past burn-in (categorical cross-entropy)
                cm = loss_mask[None, :].expand(len(mb), T)
                cden = cm.sum().clamp(min=1.0)
                if self.cat:
                    oc = OC[:, mb].transpose(0, 1)                    # [B,T,atoms]
                    closs = -(oc * F.log_softmax(c_out, dim=-1)).sum(-1)
                else:
                    v = self.value_of(c_out); closs = (v - RET[:, mb].transpose(0, 1)) ** 2
                critic_loss = C.VALUE_COEF * (closs * cm).sum() / cden
                if RT is not None and C.RUDDER_LOSS_COEF > 0.0:
                    # g(s) regression, riding on c_trunk -- zero extra forward
                    # pass (see MinGRUNet.forward_seq's `trunk` output).
                    rt_mb = RT[:, mb].transpose(0, 1); rv_mb = RV[:, mb].transpose(0, 1)
                    g_pred = self.rudder_head(c_trunk).squeeze(-1)          # [B,T]
                    rv_den = rv_mb.sum().clamp(min=1.0)
                    rudder_loss = ((g_pred - rt_mb) ** 2 * rv_mb).sum() / rv_den
                    critic_loss = critic_loss + C.RUDDER_LOSS_COEF * rudder_loss
                    rudder_losses.append(float(rudder_loss.detach()))
                if GT is not None and C.AUX_GAMMA_COEF > 0.0:
                    # Regression against a per-gamma discounted target, masked
                    # to the same rows the categorical head trains on (every
                    # row past burn-in has a legitimate -- possibly
                    # bootstrapped -- target, unlike RUDDER's ground-truth-only
                    # rows, so `cm` is the right mask here, not a validity one).
                    gt_mb = GT[:, mb].transpose(0, 1)                # [B,T,G]
                    gamma_pred = self.gamma_head(c_trunk)            # [B,T,G]
                    gamma_loss = (((gamma_pred - gt_mb) ** 2).mean(-1) * cm).sum() / cden
                    critic_loss = critic_loss + C.AUX_GAMMA_COEF * gamma_loss
                    gamma_losses.append(float(gamma_loss.detach()))
                # Both forwards ran above (actor.forward_seq / critic.forward_seq)
                # BEFORE either loss is used, and the two loss terms touch
                # DISJOINT parameter sets (actor_loss has no path to critic
                # params or vice versa) -- so one combined backward() through
                # their sum is exactly equivalent to two separate calls, just
                # one autograd traversal instead of two. Cuts backward-call
                # count in half on the hottest loop in the trainer (~60-68% of
                # wall clock per the perf/cycle log line).
                do_actor = not C.FREEZE_ACTOR and denom > 0
                self.critic_opt.zero_grad(set_to_none=True)
                if do_actor:
                    self.actor_opt.zero_grad(set_to_none=True)
                    (actor_loss + critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), C.MAX_GRAD_NORM)
                    self.actor_opt.step()
                else:
                    critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), C.MAX_GRAD_NORM)
                self.critic_opt.step()
                a_losses.append(float(actor_loss.detach()))
                c_losses.append(float(critic_loss.detach()))
                ents.append(float(((ent_a + ent_d) * fm).sum().detach() / denom))
                ent_a_list.append(float((ent_a * fm).sum().detach() / denom))
                ent_d_list.append(float((ent_d * fm).sum().detach() / denom))
                with torch.no_grad():
                    mh = (self._dur_cycles[d_mb] * fm).sum() / denom
                    hold_lens.append(float(mh))
        self.updates += 1
        self.stats = {
            "actor_loss": sum(a_losses) / len(a_losses),
            "critic_loss": sum(c_losses) / len(c_losses),
            "entropy": sum(ents) / len(ents),
            # split out of the combined H: lets a collapse be pinned to the
            # ACTION head specifically vs the duration head, rather than
            # inferring it from the combined number (the exact ambiguity that
            # left the ADAPTIVE controller's "combined H" coupling undiagnosed
            # -- see ARCHITECTURE.md's entropy-controller section).
            "action_entropy": sum(ent_a_list) / len(ent_a_list),
            "duration_entropy": sum(ent_d_list) / len(ent_d_list),
            "ent_coef": entropy_coef,
            "mean_hold": (sum(hold_lens) / len(hold_lens)) if hold_lens else 1.0,
        }
        if hca_hdivs:
            # mean |h/pi - 1| over the blend: near 0 = h agrees with pi, HCA is
            # contributing nothing; large = h and pi disagree a lot, HCA is
            # pulling hard (watch entropy alongside this -- a large hdiv with
            # entropy also crashing is the runaway-sharpening signature).
            self.stats["hca_hdiv"] = sum(hca_hdivs) / len(hca_hdivs)
        if rudder_losses:
            self.stats["rudder_loss"] = sum(rudder_losses) / len(rudder_losses)
        if gamma_losses:
            self.stats["gamma_loss"] = sum(gamma_losses) / len(gamma_losses)
        return self.stats

    # ── save / load (actor records feed play3.mjs) ─────────────────────────────
    def serialize(self) -> dict:
        return {
            "actor": self.actor.to_records(),
            "critic": self.critic.to_records(),
            "updates": self.updates,
            "ent_coef": self.ent_coef,
        }

    def load_state(self, obj: dict):
        self.actor.load_records(obj["actor"])
        self.critic.load_records(obj["critic"])
        self.updates = obj.get("updates", 0)
        # resume the adaptive controller where it left off (falls back to init)
        self.ent_coef = float(obj.get("ent_coef", self.ent_coef))
