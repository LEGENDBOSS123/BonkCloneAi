"""NFSP agent in PyTorch — port of src/rl2/agent.mjs (discrete SAC best
response + supervised average policy), same hyperparameter semantics.

Checkpoint (de)serialization emits the exact rl2 browser save layout for the
`agent` section, so saves interchange with train2.html.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .networks import MLP

# Hyperparameters mirroring src/rl2/config.mjs (sac / avg / network sections).
HIDDEN = [128, 128]
Q_LR = 3e-4
POLICY_LR = 3e-4
ALPHA_LR = 3e-4
AVG_LR = 1e-4
INIT_ALPHA = 0.2
MIN_ALPHA = 0.01
TARGET_ENTROPY_RATIO = 0.5
TAU = 0.02
POLYAK_EVERY = 4
SAC_BATCH = 1024
AVG_BATCH = 1024


class NFSPAgent:
    def __init__(self, state_dim: int, num_actions: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)

        def build():
            return MLP(state_dim, HIDDEN, num_actions).to(self.device)

        self.q1, self.q2, self.q1t, self.q2t = build(), build(), build(), build()
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        for p in [*self.q1t.parameters(), *self.q2t.parameters()]:
            p.requires_grad_(False)
        self.policy = build()
        self.avg = build()

        self.q_opt = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=Q_LR)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=POLICY_LR)
        self.avg_opt = torch.optim.Adam(self.avg.parameters(), lr=AVG_LR)
        self.log_alpha = torch.tensor(
            math.log(INIT_ALPHA), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.target_entropy = TARGET_ENTROPY_RATIO * math.log(num_actions)

        self.sac_steps = 0
        self.sl_steps = 0
        self.stats = {"q_loss": 0.0, "policy_loss": 0.0, "avg_loss": 0.0,
                      "entropy": 0.0, "alpha": INIT_ALPHA}

    # --- acting -----------------------------------------------------------------
    @torch.no_grad()
    def act_batch(self, net, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        """net: self.policy / self.avg / a snapshot MLP. states: [N, D]."""
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        logits = net(x)
        if greedy:
            return logits.argmax(dim=1).cpu().numpy()
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1).squeeze(1).cpu().numpy()

    # --- SAC best-response step ---------------------------------------------------
    def sac_step(self, batch, gamma_unused=None):
        s, a, r, s2, done, gamma_n = batch
        s = torch.from_numpy(s).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        gamma_n = torch.from_numpy(gamma_n).to(self.device)
        alpha = self.log_alpha.exp().detach()

        # n-step entropy-regularized target (exact expectation over actions).
        with torch.no_grad():
            logits2 = self.policy(s2)
            logp2 = F.log_softmax(logits2, dim=1)
            p2 = logp2.exp()
            min_qt = torch.min(self.q1t(s2), self.q2t(s2))
            v2 = (p2 * (min_qt - alpha * logp2)).sum(dim=1)
            y = r + gamma_n * (1.0 - done) * v2

        qa1 = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        qa2 = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(qa1, y) + F.mse_loss(qa2, y)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        logits = self.policy(s)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        with torch.no_grad():
            min_q = torch.min(self.q1(s), self.q2(s))
        policy_loss = (p * (alpha * logp - min_q)).sum(dim=1).mean()
        self.policy_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_opt.step()

        entropy = -(p * logp).sum(dim=1).mean().detach()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=math.log(MIN_ALPHA))

        self.sac_steps += 1
        if self.sac_steps % POLYAK_EVERY == 0:
            with torch.no_grad():
                for src, dst in ((self.q1, self.q1t), (self.q2, self.q2t)):
                    for ps, pd in zip(src.parameters(), dst.parameters()):
                        pd.mul_(1 - TAU).add_(ps, alpha=TAU)

        self.stats.update(q_loss=float(q_loss.detach()),
                          policy_loss=float(policy_loss.detach()),
                          entropy=float(entropy),
                          alpha=float(self.log_alpha.detach().exp()))

    # --- average-policy step ----------------------------------------------------------
    def sl_step(self, batch):
        s, a = batch
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        loss = F.cross_entropy(self.avg(s), a)
        self.avg_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.avg_opt.step()
        self.sl_steps += 1
        self.stats["avg_loss"] = float(loss.detach())

    # --- rl2-format save/load --------------------------------------------------------
    def serialize(self) -> dict:
        return {
            "q1": self.q1.to_records(),
            "q2": self.q2.to_records(),
            "q1t": self.q1t.to_records(),
            "q2t": self.q2t.to_records(),
            "policy": self.policy.to_records(),
            "avg": self.avg.to_records(),
            "logAlpha": float(self.log_alpha.detach()),
            "sacSteps": self.sac_steps,
            "slSteps": self.sl_steps,
        }

    def load_state(self, obj: dict):
        for name in ("q1", "q2", "q1t", "q2t", "policy", "avg"):
            getattr(self, name).load_records(obj[name])
        if isinstance(obj.get("logAlpha"), (int, float)):
            with torch.no_grad():
                self.log_alpha.fill_(float(obj["logAlpha"]))
        self.sac_steps = obj.get("sacSteps", 0)
        self.sl_steps = obj.get("slSteps", 0)
