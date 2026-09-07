"""Hindsight Credit Assignment (Harutyunyan et al. 2019, arXiv 1912.02503).

THE PROBLEM IT SOLVES
---------------------
With a terminal-only reward, vanilla PPO gives every step of an episode the
SAME advantage: A_t = R - V(s_t), and with a weak critic V ~ 0 that is just
A_t ~ +/-1 for the whole trajectory. Every action taken in a won episode is
reinforced equally, including the useless ones, and the useful and useless
contributions cancel in expectation. Measured here: kl stuck at 0.005 against a
0.03 target no matter what the learning rate or entropy coefficient did,
because the gradient carried almost no per-action information.

THE MECHANISM
-------------
Learn a HINDSIGHT DISTRIBUTION h(a | s, z): given that the episode ended in
outcome z, how likely is it that action a was taken in state s? Compare that to
what the policy would have done anyway, pi(a | s). By Bayes,

    P(z | s, a) = P(z | s) * h(a | s, z) / pi(a | s)

and since the reward here depends only on the terminal outcome,

    Q(s, a) = sum_z P(z|s) R(z) h(a|s,z)/pi(a|s)
    V(s)    = sum_z P(z|s) R(z)
    A(s, a) = sum_z P(z|s) R(z) [ h(a|s,z)/pi(a|s) - 1 ]          <-- what we use

An action is credited when knowing the outcome makes it look MORE likely than
the policy's prior -- i.e. when it was diagnostic of that outcome. This is
per-action credit extracted from a purely terminal reward, which is exactly
what is missing.

ON THE PROJECT CONSTRAINT. What is banned is HAND-INJECTED STRATEGY -- a human
deciding that "closer to the opponent" or "facing them" is good and paying for
it. Density itself was never the problem: a LEARNED dense signal carries no
human theory of how bonk is played. HCA is learned end to end from outcomes
the game already produces, and h(a|s,z) is a classifier nobody hand-specifies,
so it is admissible on the same grounds as RND novelty or RUDDER. It also
happens not to be a reward at all -- it re-weights credit for the existing
outcome reward and is return-equivalent in expectation.

WHY THIS SHOULD BEHAVE BETTER THAN THE CCA ATTEMPT
--------------------------------------------------
The earlier bonk4 experiment (hindsight.py, Mesnard et al.) conditioned a
BASELINE on a rich learned summary Phi of the whole future. Phi predicted the
outcome at 97% accuracy, so A = R - V(s, Phi) collapsed to ~0: the baseline
explained away the agent's own contribution along with the noise, and after
advantage normalisation that is pure amplified noise. Here z is just THREE
CLASSES. h cannot memorise the future through a 3-way bottleneck; the most it
can express is how much an action shifts a coarse outcome probability. The
degeneracy that killed CCA is structurally unavailable.

FAILURE MODE TO WATCH
---------------------
If h ~= pi the ratio is 1 and A_HCA ~= 0 -- HCA is contributing nothing. That is
reported every update as `hdiv` (mean |h/pi - 1|). If it sits near zero, the
classifier has found no action-outcome association and HCA is inert.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

LN_EPS = 1e-3


class HindsightHead(nn.Module):
    """h(a | s, z) and P(z | s), sharing one encoder.

    Its OWN encoder, deliberately not the actor's: h is trained on hindsight
    information, and letting those gradients into the policy's features would
    leak the outcome into the policy itself.
    """

    def __init__(self, state_dim: int, num_actions: int,
                 num_outcomes: int = 3, hidden: int | None = None):
        super().__init__()
        h = hidden or C.HCA_HIDDEN
        self.state_dim, self.num_actions = state_dim, num_actions
        self.num_outcomes = num_outcomes
        self.enc = nn.Sequential(
            nn.Linear(state_dim, h, bias=False), nn.LayerNorm(h, eps=LN_EPS),
            nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h, eps=LN_EPS), nn.ReLU())
        # h(a | s, z): the outcome enters as a one-hot appended to the features.
        self.act_head = nn.Linear(h + num_outcomes, num_actions)
        # P(z | s): the same 3-class outcome forecast the advantage needs.
        self.out_head = nn.Linear(h, num_outcomes)
        self.register_buffer(
            "outcome_reward",
            torch.tensor(C.HCA_OUTCOME_REWARD, dtype=torch.float32),
            persistent=False)

    def forward(self, s):
        f = self.enc(s)
        return f, self.out_head(f)

    def h_logits(self, f, z_onehot):
        return self.act_head(torch.cat([f, z_onehot], dim=-1))

    def all_h_logp(self, f):
        """log h(a | s, z) for EVERY z -> [n, Z, A]."""
        n, Z = f.shape[0], self.num_outcomes
        eye = torch.eye(Z, device=f.device, dtype=f.dtype)
        fz = f.unsqueeze(1).expand(n, Z, f.shape[-1])
        zz = eye.unsqueeze(0).expand(n, Z, Z)
        return F.log_softmax(self.act_head(torch.cat([fz, zz], dim=-1)), dim=-1)

    # ── the advantage ──────────────────────────────────────────────────────
    @torch.no_grad()
    def advantage(self, s, a, logp_pi):
        """A(s,a) = sum_z P(z|s) R(z) [h(a|s,z)/pi(a|s) - 1].

        `logp_pi` is log pi(a|s) under the BEHAVIOUR policy, which is what the
        collector already stored -- HCA needs the same distribution h was
        trained against, not a re-evaluated one.
        """
        f, zl = self.forward(s)
        pz = F.softmax(zl, dim=-1)                       # [n, Z]
        lh = self.all_h_logp(f)                          # [n, Z, A]
        lh_a = lh.gather(2, a.view(-1, 1, 1).expand(-1, self.num_outcomes, 1)
                         ).squeeze(-1)                    # [n, Z]
        ratio = (lh_a - logp_pi.unsqueeze(1)).exp()
        # Clipped: pi(a|s) appears in a denominator, so a rare action under a
        # sharp policy can produce an unbounded ratio and a single row would
        # dominate the batch.
        ratio = ratio.clamp(1.0 / C.HCA_RATIO_CLIP, C.HCA_RATIO_CLIP)
        adv = (pz * self.outcome_reward * (ratio - 1.0)).sum(dim=-1)
        return adv, float((ratio - 1.0).abs().mean())


class HindsightBuffer:
    """Rolling (obs, action, outcome) from FINISHED episodes.

    Reservoir-sampled at K rows per episode rather than every row: the
    classifier needs a representative sample, not the whole trajectory, and
    storing every row for 2560 envs x up to 1500 decisions would be ~0.5 GB.
    """

    def __init__(self, cap: int, state_dim: int):
        self.cap, self.n, self.w = cap, 0, 0
        self.s = np.zeros((cap, state_dim), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.z = np.zeros(cap, dtype=np.int64)

    def add(self, s, a, z):
        k = len(a)
        if k == 0:
            return
        idx = (self.w + np.arange(k)) % self.cap
        self.s[idx], self.a[idx], self.z[idx] = s, a, z
        self.w = int((self.w + k) % self.cap)
        self.n = min(self.cap, self.n + k)

    def sample(self, n, rng):
        if self.n == 0:
            return None
        i = rng.integers(0, self.n, min(n, self.n))
        return self.s[i], self.a[i], self.z[i]


def train_hindsight(head: HindsightHead, opt, buf: HindsightBuffer,
                    device, rng, epochs: int | None = None) -> dict:
    """Supervised: predict the action from (state, realised outcome), and the
    outcome from the state. Both are plain cross-entropy on completed episodes."""
    ep = epochs if epochs is not None else C.HCA_EPOCHS
    if buf.n < C.HCA_MIN_ROWS:
        return {}
    al, ol, acc = [], [], []
    for _ in range(ep):
        b = buf.sample(C.HCA_BATCH, rng)
        if b is None:
            break
        s = torch.from_numpy(b[0]).to(device)
        a = torch.from_numpy(b[1]).to(device)
        z = torch.from_numpy(b[2]).to(device)
        f, zl = head(s)
        zoh = F.one_hot(z, head.num_outcomes).to(f.dtype)
        la = head.h_logits(f, zoh)
        loss_a = F.cross_entropy(la, a)
        loss_z = F.cross_entropy(zl, z)
        opt.zero_grad()
        (loss_a + loss_z).backward()
        nn.utils.clip_grad_norm_(head.parameters(), C.MAX_GRAD_NORM)
        opt.step()
        al.append(loss_a.item()); ol.append(loss_z.item())
        with torch.no_grad():
            acc.append(float((zl.argmax(-1) == z).float().mean()))
    if not al:
        return {}
    return {"h_loss": sum(al) / len(al), "z_loss": sum(ol) / len(ol),
            "z_acc": sum(acc) / len(acc)}
