"""bonk2 PPO: clipped surrogate + GAE, categorical over the 18 joint actions,
separate actor/critic MLPs with LayerNorm (tfjs-compatible records, so actors
deploy straight into play3.mjs). All hyperparameters live in ppo6.config.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .networks import MLP

from . import config as C


def entropy_coef_at(steps: int) -> float:
    """Linear anneal START -> END over ENTROPY_DECAY_STEPS, then flat.

    Driven by ENV STEPS, not episodes: episode length varies by more than an
    order of magnitude over training, so an episode-based anneal moves at a
    rate that depends on how the agent happens to be playing.
    """
    frac = min(1.0, steps / C.ENTROPY_DECAY_STEPS)
    return C.ENTROPY_COEF_START + (C.ENTROPY_COEF_END - C.ENTROPY_COEF_START) * frac


def duration_entropy_coef_at(steps: int) -> float:
    """Separate anneal for the FiGAR duration head (see config.DURATION_ENTROPY_*).
    The duration head collapses under the action head's coef, so it gets its own,
    stronger schedule to keep long holds sampled and exploration alive."""
    frac = min(1.0, steps / C.ENTROPY_DECAY_STEPS)
    return (C.DURATION_ENTROPY_COEF_START
            + (C.DURATION_ENTROPY_COEF_END - C.DURATION_ENTROPY_COEF_START) * frac)


def intrinsic_coef_at(steps: int) -> float:
    """beta: the weight of the ensemble-disagreement intrinsic advantage,
    annealed START -> END over INTRINSIC_DECAY_STEPS so the exploration bonus
    fades as the map gets mapped out (ppo8)."""
    if not getattr(C, "ENSEMBLE_DISAGREEMENT", False):
        return 0.0
    frac = min(1.0, steps / C.INTRINSIC_DECAY_STEPS)
    return C.INTRINSIC_COEF_START + (C.INTRINSIC_COEF_END - C.INTRINSIC_COEF_START) * frac


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
    def __init__(self, state_dim: int, num_actions: int, device: str = "cpu",
                 atoms=None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        # Two heads packed into one output vector: [0:num_actions] = joint action
        # logits, [num_actions:] = duration logits. One MLP keeps records/deploy
        # simple; the split happens wherever the policy is read.
        self.n_dur = C.NUM_DURATIONS
        self.actor = MLP(state_dim, C.HIDDEN,
                         num_actions + self.n_dur).to(self.device)
        # `atoms` lets an EXPLOITER use its own outcome values (a draw is worth
        # 0 to it, not -1) while sharing all other machinery. Must be set
        # BEFORE the critic, which is sized from it.
        self.atom_vals = tuple(atoms if atoms is not None else C.CRITIC_ATOMS)
        self.atoms = torch.tensor(self.atom_vals, dtype=torch.float32,
                                  device=self.device)
        # One logit per atom when categorical, else a scalar value.
        self.cat = bool(C.CRITIC_CATEGORICAL)
        self.critic = MLP(state_dim, C.HIDDEN,
                          len(self.atom_vals) if self.cat else 1).to(self.device)
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
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=C.ACTOR_LR)
        # aux_head rides with the critic: same trunk, same learning rate, and
        # crucially NOT in the actor's optimizer.
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.aux_head.parameters())
            + list(self.time_head.parameters()),
            lr=C.CRITIC_LR)
        # ── ppo8: disagreement ensemble + intrinsic value head ─────────────────
        # ensemble = ENSEMBLE_N small scalar critics that regress the SAME
        # extrinsic return; their VARIANCE at a state is the intrinsic reward.
        # intrinsic_head reads the main critic's trunk (a head "on the main
        # critic", per the design) but trains on DETACHED features so it can
        # never perturb the extrinsic value the policy actually cares about.
        # Only phase 3 (LEAGUE_ENABLED) uses the ensemble; phases 1 & 2 are a
        # focused curriculum / critic warmup where intrinsic exploration hurts,
        # so they run a single critic and never build the ensemble at all.
        self.use_ensemble = (bool(getattr(C, "ENSEMBLE_DISAGREEMENT", False))
                             and bool(getattr(C, "LEAGUE_ENABLED", True)))
        if self.use_ensemble:
            self.ensemble = torch.nn.ModuleList([
                MLP(state_dim, C.ENSEMBLE_HIDDEN, 1).to(self.device)
                for _ in range(C.ENSEMBLE_N)])       # distinct init per member
            self.ensemble_opt = torch.optim.Adam(self.ensemble.parameters(),
                                                 lr=C.ENSEMBLE_LR)
            self.intrinsic_head = torch.nn.Linear(C.HIDDEN[-1], 1).to(self.device)
            torch.nn.init.zeros_(self.intrinsic_head.bias)
            torch.nn.init.normal_(self.intrinsic_head.weight, 0.0, 0.01)
            self.intrinsic_opt = torch.optim.Adam(self.intrinsic_head.parameters(),
                                                  lr=C.CRITIC_LR)
            self._rint_std = None        # running std of r_int for normalization
        self.updates = 0
        self.stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    # ── ppo8: ensemble-disagreement intrinsic reward ────────────────────────────
    @torch.no_grad()
    def ensemble_disagreement(self, states: np.ndarray) -> np.ndarray:
        """RAW intrinsic reward = std across the ensemble's scalar value
        estimates. High where the members disagree = the agent hasn't figured
        this state out. Normalization/clipping happens in normalize_intrinsic."""
        x = torch.from_numpy(np.ascontiguousarray(states)).float().to(self.device)
        preds = torch.stack([m(x).squeeze(-1) for m in self.ensemble], 0)  # [K, N]
        return preds.std(0).cpu().numpy().astype(np.float32)

    def normalize_intrinsic(self, r_int: np.ndarray) -> np.ndarray:
        """Scale r_int to ~unit std via a running estimate (so beta is portable,
        not tied to the raw disagreement magnitude), then clip outliers."""
        if not getattr(C, "INTRINSIC_NORMALIZE", True):
            return np.clip(r_int, 0.0, C.INTRINSIC_REWARD_CLIP)
        batch_std = float(r_int.std())
        # EMA of the std; init to the first batch so early steps aren't blown up
        if self._rint_std is None:
            self._rint_std = max(batch_std, 1e-6)
        else:
            self._rint_std = 0.99 * self._rint_std + 0.01 * batch_std
        out = r_int / (self._rint_std + 1e-8)
        return np.clip(out, 0.0, C.INTRINSIC_REWARD_CLIP).astype(np.float32)

    @torch.no_grad()
    def intrinsic_value(self, states: np.ndarray) -> np.ndarray:
        """V_int(s): the intrinsic-return baseline (its own head on the critic
        trunk). Used to GAE the intrinsic reward stream."""
        x = torch.from_numpy(np.ascontiguousarray(states)).float().to(self.device)
        return self.intrinsic_head(self.critic.trunk(x)).squeeze(-1).cpu().numpy().astype(np.float32)

    def train_intrinsic(self, states: np.ndarray, ext_returns: np.ndarray,
                        int_returns: np.ndarray):
        """Fit the ensemble to the EXTRINSIC return (bootstrap masks give the
        members diverse data -> real disagreement) and the intrinsic head to the
        INTRINSIC return (on detached trunk features). Both are plain MSE."""
        if not self.use_ensemble:
            return {}
        dev = self.device
        n = states.shape[0]; n -= n % 2048
        if n <= 0:
            return {}
        x = torch.from_numpy(np.ascontiguousarray(states[:n])).float().to(dev)
        er = torch.from_numpy(np.ascontiguousarray(ext_returns[:n])).float().to(dev)
        ir = torch.from_numpy(np.ascontiguousarray(int_returns[:n])).float().to(dev)
        # ensemble: each member sees a random ~BOOTSTRAP_P subset of rows
        self.ensemble_opt.zero_grad(set_to_none=True)
        ens_loss = 0.0
        for m in self.ensemble:
            mask = torch.rand(n, device=dev) < C.ENSEMBLE_BOOTSTRAP_P
            if not bool(mask.any()):
                continue
            pred = m(x[mask]).squeeze(-1)
            loss = F.mse_loss(pred, er[mask])
            loss.backward()
            ens_loss += float(loss.detach())
        self.ensemble_opt.step()
        # intrinsic head: detached trunk features so the extrinsic critic is safe
        with torch.no_grad():
            feats = self.critic.trunk(x)
        self.intrinsic_opt.zero_grad(set_to_none=True)
        v_int = self.intrinsic_head(feats).squeeze(-1)
        iv_loss = F.mse_loss(v_int, ir)
        iv_loss.backward()
        self.intrinsic_opt.step()
        return {"ens_loss": ens_loss / max(1, C.ENSEMBLE_N),
                "vint_loss": float(iv_loss.detach())}

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
        la, ld = self._split(self.actor(x))
        lpa, lpd = F.log_softmax(la, 1), F.log_softmax(ld, 1)
        a = torch.multinomial(lpa.exp(), 1).squeeze(1)
        d = torch.multinomial(lpd.exp(), 1).squeeze(1)
        logp = (lpa.gather(1, a.unsqueeze(1)) + lpd.gather(1, d.unsqueeze(1))).squeeze(1)
        v = self.value_of(self.critic(x))
        return (a.cpu().numpy(), d.cpu().numpy(),
                logp.cpu().numpy(), v.cpu().numpy())

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
        for _ in range(C.EPOCHS):
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
                s2 = ratio.clamp(1 - C.CLIP_EPS, 1 + C.CLIP_EPS) * adv[idx]
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
                actor_loss = ((surr * fm).sum() - ent_bonus) / denom
                entropy = (ent_row * fm).sum() / denom
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
        }
        return self.stats

    # ── save / load (actor records feed play3.mjs) ─────────────────────────────
    def serialize(self) -> dict:
        return {
            "actor": self.actor.to_records(),
            "critic": self.critic.to_records(),
            "updates": self.updates,
        }

    def load_state(self, obj: dict):
        self.actor.load_records(obj["actor"])
        self.critic.load_records(obj["critic"])
        self.updates = obj.get("updates", 0)
