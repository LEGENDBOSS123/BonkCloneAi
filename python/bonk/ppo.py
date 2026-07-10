"""PPO agent (PyTorch) over the joint 18-action space — the rl1 algorithm
(clipped surrogate, GAE, separate actor/critic with LayerNorm, mirror augment)
on the rl2 observation/action interface, so checkpoints slot into the same
tooling (play.py --net actor, future play2.mjs).

Policy is categorical over 18 joint actions (softmax), not 5 Bernoullis:
cleaner math, no contradictory key combos, and matches the NFSP nets.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .networks import MLP

HIDDEN = [256, 256]
ACTOR_LR = 2.5e-4
CRITIC_LR = 3e-4
CLIP_EPS = 0.2
EPOCHS = 4
MINIBATCH = 8192
# Entropy coefficient anneals linearly from START to END over the first
# ENTROPY_DECAY_EPISODES episodes, then holds at END: explore hard while
# self-play is still discovering strategies, sharpen for the rest of the run.
ENTROPY_COEF_START = 0.2
ENTROPY_COEF_END = 0.01
ENTROPY_DECAY_EPISODES = 2_500_000
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
NORMALIZE_ADV = True
CLIP_VALUE_LOSS = True


def entropy_coef_at(episode: int) -> float:
    """Linear anneal START -> END over ENTROPY_DECAY_EPISODES, then flat."""
    frac = min(1.0, episode / ENTROPY_DECAY_EPISODES)
    return ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * frac


class PPOAgent:
    def __init__(self, state_dim: int, num_actions: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.actor = MLP(state_dim, HIDDEN, num_actions).to(self.device)
        self.critic = MLP(state_dim, HIDDEN, 1).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.updates = 0
        self.stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    # --- acting -----------------------------------------------------------------
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
        """Actions only, from an arbitrary actor (snapshot opponents)."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        logits = net(x)
        if greedy:
            return logits.argmax(dim=1).cpu().numpy()
        return torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(1).cpu().numpy()

    # --- PPO update ---------------------------------------------------------------
    def update(self, rollout: dict, entropy_coef: float = ENTROPY_COEF_END):
        """rollout: numpy arrays states [N,D], actions [N], logps [N],
        values [N], advantages [N], returns [N]. `entropy_coef` comes from the
        trainer's schedule (entropy_coef_at)."""
        dev = self.device
        states = torch.from_numpy(rollout["states"]).to(dev)
        actions = torch.from_numpy(rollout["actions"]).to(dev)
        old_logp = torch.from_numpy(rollout["logps"]).to(dev)
        old_v = torch.from_numpy(rollout["values"]).to(dev)
        returns = torch.from_numpy(rollout["returns"]).to(dev)
        adv = torch.from_numpy(rollout["advantages"]).to(dev)
        if NORMALIZE_ADV:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = states.shape[0]
        a_losses, c_losses, ents = [], [], []
        for _ in range(EPOCHS):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, MINIBATCH):
                idx = perm[s:s + MINIBATCH]
                mb_s = states[idx]

                logits = self.actor(mb_s)
                logp_all = F.log_softmax(logits, dim=1)
                logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]
                entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()
                actor_loss = -torch.min(s1, s2).mean() - entropy_coef * entropy
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_opt.step()

                v = self.critic(mb_s).squeeze(1)
                if CLIP_VALUE_LOSS:
                    v_clip = old_v[idx] + (v - old_v[idx]).clamp(-CLIP_EPS, CLIP_EPS)
                    critic_loss = torch.max((v - returns[idx]) ** 2,
                                            (v_clip - returns[idx]) ** 2).mean()
                else:
                    critic_loss = F.mse_loss(v, returns[idx])
                critic_loss = VALUE_COEF * critic_loss
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
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

    # --- save / load (rl2-style JSON; actor records feed play2.mjs later) -----------
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
