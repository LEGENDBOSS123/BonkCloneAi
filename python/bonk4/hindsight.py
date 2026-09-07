"""Counterfactual Credit Assignment: a FUTURE-conditional baseline.

    Mesnard et al. 2021, arxiv 2011.09464 (model-free CCA)
    Harutyunyan et al. 2019, arxiv 1912.02503 (hindsight credit assignment)

Everything else in ppo4 tries to make V forecast the outcome better from the
prefix. Measured, that is capped by information rather than capacity: in
symmetric self-play P(win) = 0.5 at t=0 by construction, the critic reports it
honestly (|V| ~ 0.13 for the first 80% of an episode, ev = 0.458 only in the
last 5%), and no architecture manufactures predictability that is not there.

CCA changes the baseline's INFORMATION SET instead of its capacity:

    A_t = R - V(s_t)          baseline sees the past
    A_t = R - V(s_t, Phi_t)   baseline also sees what actually happened next

Phi_t summarises the trajectory AFTER t, so it explains away the randomness the
agent did not control — above all the opponent. What is left in A_t is much
closer to this action's own contribution, which is why the paper's baselines are
provably low variance. Hindsight is a different information set, not a smarter
forward model, so it is not bound by the ceiling above.

THE BIAS PROBLEM. A baseline may condition on anything except the agent's own
action; otherwise it is action-dependent and the policy gradient is biased. But
s_{t+1} depends on a_t, so Phi_t leaks action information no matter what. Two
defences, both here:

  1. structural masking — the own-action block obs[28:32] is zeroed out of the
     backward encoder's input. Exact, and only possible because this obs is
     structured rather than pixels.
  2. an adversarial independence penalty — a head predicts a_t from Phi_t and a
     gradient-reversal layer trains Phi to make that fail. This is the paper's
     mechanism for the residual leakage masking cannot reach.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C

LN_EPS = 1e-3


class _GradReverse(torch.autograd.Function):
    """Identity forwards, negated gradient backwards.

    The adversary minimises its own loss (it genuinely tries to recover a_t);
    the encoder sees the sign flipped, so the SAME loss trains Phi to destroy
    the action information. One backward pass, no alternating optimisation.
    """

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float):
    return _GradReverse.apply(x, lam)


class HindsightBaseline(nn.Module):
    """Backward GRU over the future + a future-conditional outcome head."""

    def __init__(self, in_dim: int, z_dim: int, num_actions: int,
                 hidden: int | None = None):
        super().__init__()
        h = hidden or C.HCA_HIDDEN
        self.hidden, self.in_dim, self.z_dim = h, in_dim, z_dim

        # Its OWN encoder, deliberately not the shared one: the shared encoder
        # feeds the policy, and letting hindsight gradients into it would leak
        # future information into the actor's features.
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h, bias=False), nn.LayerNorm(h, eps=LN_EPS), nn.ReLU())
        self.back = nn.GRUCell(h, h)
        # Baseline reads the present (z_t, from the shared encoder, detached)
        # alongside the future summary.
        self.value = nn.Sequential(
            nn.Linear(z_dim + h, h), nn.ReLU(), nn.Linear(h, C.NUM_OUTCOMES))
        # Adversary: must NOT be able to recover a_t from Phi_t.
        self.adversary = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, num_actions))
        self.register_buffer("outcome_reward",
                             torch.tensor(C.OUTCOME_REWARD, dtype=torch.float32),
                             persistent=False)
        # Own-action bits are masked out of the backward input as a hard prior.
        m = torch.ones(in_dim)
        lo, hi = C.HCA_MASK_SLICE
        if 0 <= lo < hi <= in_dim:
            m[lo:hi] = 0.0
        self.register_buffer("in_mask", m, persistent=False)

    def phi(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Phi_t = summary of steps t+1..T, from a BACKWARD pass.

        obs/mask are [L, n, ...] padded episodes. Phi_t deliberately excludes
        step t itself: the baseline already sees the present through z_t, and
        including o_t would hand it the most direct trace of a_t there is.
        """
        L, n, _ = obs.shape
        x = self.enc(obs * self.in_mask)
        hcur = torch.zeros(n, self.hidden, device=obs.device, dtype=x.dtype)
        out = [None] * L
        for t in range(L - 1, -1, -1):
            out[t] = hcur                      # summary of t+1.. BEFORE folding t
            m = mask[t].unsqueeze(-1).to(x.dtype)
            hcur = self.back(x[t], hcur) * m + hcur * (1.0 - m)
        return torch.stack(out)

    def forward(self, z: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor):
        """-> (outcome logits [L,n,3], adversary logits [L,n,A])."""
        p = self.phi(obs, mask)
        logits = self.value(torch.cat([z, p], dim=-1))
        adv = self.adversary(grad_reverse(p, C.HCA_ADV_COEF))
        return logits, adv

    def value_of(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1) @ self.outcome_reward

    def to_records(self) -> dict:
        return {"format": "hindsight-v1", "hidden": self.hidden,
                "inDim": self.in_dim, "zDim": self.z_dim,
                "tensors": {k: v.detach().cpu().numpy().tolist()
                            for k, v in self.state_dict().items()}}

    def load_records(self, obj: dict):
        sd = {k: torch.tensor(v) for k, v in obj["tensors"].items()}
        own = self.state_dict()
        keep = {k: v for k, v in sd.items()
                if k in own and v.shape == own[k].shape}
        if len(keep) != len(own):
            print(f"note: hindsight baseline partially reinitialised "
                  f"({len(own) - len(keep)} tensors)")
            for k in own:
                keep.setdefault(k, own[k])
        self.load_state_dict(keep)
