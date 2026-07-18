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
    """Linear anneal START -> END over ENTROPY_DECAY_EPISODES, then flat."""
    frac = min(1.0, episode / C.ENTROPY_DECAY_EPISODES)
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
        self.actor = MLP(state_dim, C.HIDDEN, num_actions).to(self.device)
        self.critic = MLP(state_dim, C.HIDDEN, 1).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=C.ACTOR_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=C.CRITIC_LR)
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
        v = self.critic(x).squeeze(1)
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
        a_losses, c_losses, ents = [], [], []
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
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               C.MAX_GRAD_NORM)
                self.actor_opt.step()

                v = self.critic(mb_s).squeeze(1)
                if C.CLIP_VALUE_LOSS:
                    v_clip = old_v[idx] + (v - old_v[idx]).clamp(-C.CLIP_EPS,
                                                                 C.CLIP_EPS)
                    critic_loss = torch.max((v - returns[idx]) ** 2,
                                            (v_clip - returns[idx]) ** 2).mean()
                else:
                    critic_loss = F.mse_loss(v, returns[idx])
                critic_loss = C.VALUE_COEF * critic_loss
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
