"""Python port of src/rl2/lag_env.mjs: decision-level env with input lag,
joint discrete actions, normalized 30-dim observations, and terminal rewards.

Observation layout (identical to the JS LagEnv — see lag_env.mjs header):
  [ 0- 9] self  : x, y, vx, vy, heavyValue, up, down, left, right, heavyKey
  [10-19] opp   : same
  [20-23] rel   : dx, dy, dvx, dvy
  [24-28] pend  : own last DECIDED action bits [up, down, left, right, heavy]
  [29]    time  : episode_steps / MAX_EPISODE_STEPS

The lag mechanism is generalized to a decision log: a decision made at tick t
governs physics from tick t + INPUT_LAG. At the JS decision cadence (every
ACTION_REPEAT ticks) this reduces exactly to the JS prev/last behavior; it also
supports per-tick human input in the playground.
"""

import random

import numpy as np

from . import config as C
from .sim import BonkSim, IDLE_KEYS, index_to_keys

IDLE = 0


class LagEnv:
    def __init__(self, sim: BonkSim):
        self.sim = sim
        self.lag = C.INPUT_LAG
        self.max_steps = C.MAX_EPISODE_STEPS
        self.state_dim = C.STATE_DIM
        # Per seat: list of (decided_tick, action); pruned as entries expire.
        self.decisions = [[], []]
        self.episode_steps = 0

    def reset(self, randomize_spawns=True):
        self.sim.reset(randomize_spawns)
        self.decisions = [[], []]
        self.episode_steps = 0
        # Per-episode lag randomization (see config.INPUT_LAG_RANDOM): the agent
        # can't observe the draw, so it must learn timing robust to the range.
        if getattr(C, "INPUT_LAG_RANDOM", False):
            self.lag = random.randint(C.INPUT_LAG_MIN, C.INPUT_LAG_MAX)

    def set_decision(self, seat: int, action: int):
        self.decisions[seat].append((self.episode_steps, action))

    def _applied_index(self, seat: int) -> int:
        """Latest decision old enough to have taken effect (t_decided <= now-lag)."""
        cutoff = self.episode_steps - self.lag
        log = self.decisions[seat]
        applied = IDLE
        keep_from = 0
        for i, (t, a) in enumerate(log):
            if t <= cutoff:
                applied = a
                keep_from = i
            else:
                break
        # Prune everything before the applied entry (it can never win again).
        if keep_from > 0:
            del log[:keep_from]
        return applied

    def _last_decided(self, seat: int) -> int:
        log = self.decisions[seat]
        return log[-1][1] if log else IDLE

    def applied_actions(self):
        return [self._applied_index(0), self._applied_index(1)]

    def tick(self):
        k0 = index_to_keys(self._applied_index(0))
        k1 = index_to_keys(self._applied_index(1))
        self.sim.step(k0, k1)
        self.episode_steps += 1

        dead = [self.sim.is_dead(0), self.sim.is_dead(1)]
        timeout = not any(dead) and self.episode_steps >= self.max_steps
        done = any(dead) or timeout

        rewards = [0.0, 0.0]
        if done:
            if timeout or (dead[0] and dead[1]):
                rewards = [C.DRAW_REWARD, C.DRAW_REWARD]
            elif dead[1]:
                rewards = [C.WIN_REWARD, C.LOSS_REWARD]
            else:
                rewards = [C.LOSS_REWARD, C.WIN_REWARD]
        return {"rewards": rewards, "done": done, "dead": dead, "timeout": timeout}

    def decision_state(self, seat: int) -> np.ndarray:
        s = self.sim.players[seat]
        o = self.sim.players[1 - seat]
        out = np.zeros(C.STATE_DIM, dtype=np.float32)

        def block(off, player, applied_idx):
            k = index_to_keys(applied_idx)
            x, y = player.pos
            vx, vy = player.vel
            out[off] = x * C.POS_SCALE
            out[off + 1] = y * C.POS_SCALE
            out[off + 2] = vx * C.VEL_SCALE
            out[off + 3] = vy * C.VEL_SCALE
            out[off + 4] = player.heavy_power * C.HEAVY_SCALE if k["heavy"] else 0.0
            out[off + 5] = 1.0 if k["up"] else 0.0
            out[off + 6] = 1.0 if k["down"] else 0.0
            out[off + 7] = 1.0 if k["left"] else 0.0
            out[off + 8] = 1.0 if k["right"] else 0.0
            out[off + 9] = 1.0 if k["heavy"] else 0.0

        block(0, s, self._applied_index(seat))
        block(10, o, self._applied_index(1 - seat))

        sx, sy = s.pos
        ox, oy = o.pos
        svx, svy = s.vel
        ovx, ovy = o.vel
        out[20] = (ox - sx) * C.POS_SCALE
        out[21] = (oy - sy) * C.POS_SCALE
        out[22] = (ovx - svx) * C.VEL_SCALE
        out[23] = (ovy - svy) * C.VEL_SCALE

        pk = index_to_keys(self._last_decided(seat))
        out[24] = 1.0 if pk["up"] else 0.0
        out[25] = 1.0 if pk["down"] else 0.0
        out[26] = 1.0 if pk["left"] else 0.0
        out[27] = 1.0 if pk["right"] else 0.0
        out[28] = 1.0 if pk["heavy"] else 0.0

        out[29] = min(1.0, self.episode_steps / self.max_steps)
        return out
