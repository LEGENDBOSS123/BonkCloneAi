"""Movement skill module — the transplantable CORE.

The agent has learned not to fall off and nothing else: at real spawns it stands
still, because crossing tens of metres of lethal gap by random action search
never pays off before the episode ends. Navigation is a dense, easy, SINGLE-AGENT
problem hiding underneath a sparse two-player one, so learn it separately and
hand the result to the policy.

Shape (d_emb = 64; the task is simple and a bigger core just gives PPO scratch
capacity to abuse):

    MovementNet          own_state[12] ++ goal[4]
                              L_in   16 -> 64          <- discarded at transplant
                         +--- CORE  64 -> 64 -> 64 ---+ <- TRANSPLANTED
                              L_out  64 -> 18          <- discarded at transplant

At transplant the actor learns `proj_in` into the CORE's embedding space and
reads the CORE's output as extra features. Only the middle survives.

TRAINING: goal-conditioned behaviour cloning on HINDSIGHT-RELABELLED self-play
data. Take any (s_t, s_{t+k}) from a trajectory already collected, call
s_{t+k}'s position/velocity the goal, and train the net to output the action
that was actually taken. Every trajectory is a correct demonstration of reaching
wherever it happened to end up, so labels are free, always correct, and need no
extra environment interaction. Nothing here knows anything about bonk beyond
"go to a place", so it passes the constraint test.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

LN_EPS = 1e-3
GOAL_DIM = 4        # target x, y, vx, vy  (obs[0:4] of a later step)
OWN_DIM = 12        # obs[0:12] is the agent's own block


class MovementCore(nn.Module):
    """d_emb -> d_emb. The only part that survives transplant."""

    def __init__(self, d_emb: int | None = None, hidden: int | None = None):
        super().__init__()
        d = d_emb or C.MOVE_EMB
        h = hidden or C.MOVE_HIDDEN
        self.d_emb = d
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h, eps=LN_EPS), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h, eps=LN_EPS), nn.ReLU(),
            nn.Linear(h, d), nn.LayerNorm(d, eps=LN_EPS))

    def forward(self, u):
        return self.net(u)


class MovementNet(nn.Module):
    """L_in + CORE + L_out. Only exists to give the CORE a training signal."""

    def __init__(self, num_actions: int, core: MovementCore | None = None):
        super().__init__()
        self.core = core or MovementCore()
        d = self.core.d_emb
        self.l_in = nn.Sequential(
            nn.Linear(OWN_DIM + GOAL_DIM, d), nn.LayerNorm(d, eps=LN_EPS), nn.ReLU())
        self.l_out = nn.Linear(d, num_actions)
        # Value head for dense goal-reaching pre-training. Discarded at
        # transplant along with l_in / l_out.
        self.v_out = nn.Linear(d, 1)
        # Running statistics of what L_in produces. The actor's learned proj_in
        # is normalised to these at transplant: without it the actor can drive
        # the CORE anywhere in R^d, including regions where its structure means
        # nothing, and the CORE degenerates into an arbitrary fixed nonlinearity.
        self.register_buffer("emb_mean", torch.zeros(d))
        self.register_buffer("emb_var", torch.ones(d))
        self.register_buffer("emb_n", torch.zeros(1))

    def forward(self, own, goal):
        u = self.l_in(torch.cat([own, goal], dim=-1))
        if self.training:
            with torch.no_grad():
                flat = u.reshape(-1, u.shape[-1])
                n = self.emb_n + flat.shape[0]
                w = flat.shape[0] / n
                self.emb_mean.mul_(1 - w).add_(w * flat.mean(0))
                self.emb_var.mul_(1 - w).add_(w * flat.var(0, unbiased=False))
                self.emb_n.copy_(n)
        c = self.core(u)
        return self.l_out(c), c


def gcbc_pairs(pool, n_pairs: int, k_lo: int = 4, k_hi: int = 40, rng=None):
    """Sample hindsight (own_state, goal, action) triples from finished episodes.

    goal is the agent's OWN position/velocity k steps later, so the label a_t is
    by construction an action that made progress toward it.
    """
    rng = rng or np.random.default_rng()
    if not pool.eps:
        return None
    offs = np.array([e[0] for e in pool.eps]); lens = np.array([e[1] for e in pool.eps])
    ok = lens > k_lo + 1
    if not ok.any():
        return None
    offs, lens = offs[ok], lens[ok]
    ei = rng.integers(0, len(lens), n_pairs)
    k = rng.integers(k_lo, k_hi + 1, n_pairs)
    k = np.minimum(k, lens[ei] - 1)
    t = (rng.random(n_pairs) * (lens[ei] - k)).astype(np.int64)
    src = offs[ei] + t
    dst = src + k
    return (pool.s[src, :OWN_DIM].copy(),
            pool.s[dst, :GOAL_DIM].copy(),
            pool.a[src].copy())


def gcbc_loss(net: MovementNet, own, goal, act):
    logits, _c = net(own, goal)
    return F.cross_entropy(logits, act)


def goal_reward(pos, vel, goal):
    """Dense: how close is the agent to the target state right now.

    Negative normalised distance in position plus a weighted velocity term, so
    the gradient exists at every step regardless of whether the goal is ever
    reached. Purely a motor objective — a person who has never played bonk can
    write "get to this point moving at this speed", so it passes the constraint.
    """
    dp = np.linalg.norm(pos - goal[:2], axis=-1) / C.MOVE_POS_W
    dv = np.linalg.norm(vel - goal[2:4], axis=-1) / C.MOVE_POS_W
    return -(dp + C.MOVE_VEL_W * dv)
