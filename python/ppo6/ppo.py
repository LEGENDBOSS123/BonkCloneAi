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
        self.actor = MLP(state_dim, C.HIDDEN, num_actions).to(self.device)
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
        self.updates = 0
        self.stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    # ── acting ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act_batch(self, states: np.ndarray):
        """Sample actions for the learner: returns (actions, logps, values)."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        logits = self.actor(x)
        logp_all = F.log_softmax(logits, dim=1)
        a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
        logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        v = self.value_of(self.critic(x))
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
    def value_of(self, out):
        """critic output -> scalar V. Categorical: a convex combination of the
        atoms, hence bounded by construction."""
        if not self.cat:
            return out.squeeze(-1)
        return torch.softmax(out, dim=-1) @ self.atoms

    def update(self, rollout: dict, entropy_coef: float = C.ENTROPY_COEF_END,
               shaping_w: float = 0.0):
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

                logits = self.actor(mb_s)
                logp_all = F.log_softmax(logits, dim=1)
                logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - C.CLIP_EPS, 1 + C.CLIP_EPS) * adv[idx]
                entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()
                actor_loss = -torch.min(s1, s2).mean() - entropy_coef * entropy
                if not C.FREEZE_ACTOR:
                    self.actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               C.MAX_GRAD_NORM)
                if not C.FREEZE_ACTOR:
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
