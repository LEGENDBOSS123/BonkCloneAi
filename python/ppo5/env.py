"""Slime Volleyball with ppo4's REWARD SHAPE: terminal-only, 3 outcomes.

The control experiment for ppo4. Slime Volleyball is David Ha's self-play
benchmark — simultaneous, physics-driven, 12-dim continuous state, discrete
actions, ~600-step matches with both seats symmetric. Structurally it is the
closest prebuilt thing to bonk there is.

One change: the native env pays +1/-1 on every point. That is DENSER than bonk,
which would let the critic off easy, so points are suppressed and the match pays
out once at the end:

    WIN   finished ahead on points
    LOSS  finished behind
    DRAW  level at the time limit

Now the critic faces bonk's exact job — predict which of three ways a ~600-step
match ends, from a prefix, with nothing to learn from until it does.
"""

from __future__ import annotations

import itertools

import numpy as np
import slimevolleygym

OUT_WIN, OUT_LOSS, OUT_DRAW = 0, 1, 2
STATE_DIM = 12
# MultiBinary(3) = [left, right, jump]; all 8 combinations as joint actions,
# the same "one discrete index over button combos" scheme bonk uses.
ACTIONS = list(itertools.product([0, 1], repeat=3))
NUM_ACTIONS = len(ACTIONS)


class VecSlimeVolley:
    """E independent matches, both seats driven externally (self-play).

    step() returns per-seat done/outcome. Seat 0 is the right player (whose obs
    the env returns natively); seat 1 is the left player, whose mirrored obs
    arrives in info["otherObs"]. Their outcomes are exact opposites.
    """

    def __init__(self, n_envs: int, seed: int = 0, t_limit: int = 3000):
        self.E = n_envs
        self.envs = []
        for i in range(n_envs):
            e = slimevolleygym.SlimeVolleyEnv()
            e.survival_bonus = False
            e.t_limit = t_limit
            e.seed(seed + i)
            self.envs.append(e)
        self.o0 = np.zeros((n_envs, STATE_DIM), dtype=np.float32)
        self.o1 = np.zeros((n_envs, STATE_DIM), dtype=np.float32)
        self.score = np.zeros(n_envs, dtype=np.int64)
        self.reset_all()

    def _set_obs(self, i, o, info):
        self.o0[i] = np.asarray(o, dtype=np.float32)
        self.o1[i] = np.asarray(info["otherObs"], dtype=np.float32)

    def reset_all(self):
        for i, e in enumerate(self.envs):
            o = e.reset()
            self.o0[i] = np.asarray(o, dtype=np.float32)
            # before the first step there is no info; the left obs is the
            # mirror of the right one at the symmetric start state
            self.o1[i] = np.asarray(o, dtype=np.float32)
            self.score[i] = 0

    def reset_one(self, i):
        o = self.envs[i].reset()
        self.o0[i] = np.asarray(o, dtype=np.float32)
        self.o1[i] = np.asarray(o, dtype=np.float32)
        self.score[i] = 0

    def step(self, a0: np.ndarray, a1: np.ndarray):
        """a0/a1 are discrete indices into ACTIONS. Returns (done, out0, out1)."""
        done = np.zeros(self.E, dtype=bool)
        out0 = np.full(self.E, -1, dtype=np.int64)
        out1 = np.full(self.E, -1, dtype=np.int64)
        for i, e in enumerate(self.envs):
            o, r, d, info = e.step(list(ACTIONS[int(a0[i])]),
                                   list(ACTIONS[int(a1[i])]))
            self._set_obs(i, o, info)
            self.score[i] += int(r)      # +1 when seat 0 scores, -1 when scored on
            if d:
                s = self.score[i]
                c0 = OUT_WIN if s > 0 else (OUT_LOSS if s < 0 else OUT_DRAW)
                c1 = OUT_LOSS if s > 0 else (OUT_WIN if s < 0 else OUT_DRAW)
                done[i] = True
                out0[i], out1[i] = c0, c1
        return done, out0, out1
