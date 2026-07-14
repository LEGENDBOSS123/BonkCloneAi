"""bonk2 LagEnv: decision-level env with input lag, joint discrete actions,
normalized 34-dim observations, and sparse terminal rewards.

Same lag mechanics as v1 (bonk.env2): a decision made at tick t governs physics
from tick t + lag, with lag randomized per episode as the sim-to-real bridge.

v2 obs change — heavy-meter memory. The raw heavyValue is only observable while
the heavy key is held (masked to 0 otherwise), but the meter silently
regenerates at 5/tick while released. v1's obs therefore lost information: a
player who released at 30% and re-pressed 100 ticks later shows up exactly like
one who released at 80%. v2 adds, per player, the last heavy value seen while
the key was held and how many ticks ago that was (normalized by the 200-tick
full-regen span, capped at 1) — both derived purely from what is observable, so
the obs stays map-agnostic and works identically in the browser (play3.mjs).

Observation layout (STATE_DIM = 34):
  [ 0-11] self : x, y, vx, vy, heavyValue(masked), up, down, left, right,
                 heavyKey, lastHeavySeen, ticksSinceSeen/200 (cap 1)
  [12-23] opp  : same 12
  [24-27] rel  : dx, dy, dvx, dvy
  [28-32] pend : own last DECIDED action bits [up, down, left, right, heavy]
  [33]    time : episode_steps / MAX_EPISODE_STEPS
"""

import random

import numpy as np

from bonk.sim import BonkSim, index_to_keys

from . import config as C

IDLE = 0


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Horizontal mirror of a 34-dim obs (x/vx negate, left/right bits swap).
    Heavy features are symmetric and pass through unchanged."""
    out = obs.copy()
    out[C.MIRROR_NEGATE] *= -1
    for a, b in C.MIRROR_SWAPS:
        out[a], out[b] = out[b], out[a]
    return out


def mirror_action(index: int) -> int:
    """Swap left/right in the joint-action index; up/down/heavy unchanged."""
    lr, rest = index // 6, index % 6
    return (2 if lr == 1 else 1 if lr == 2 else 0) * 6 + rest


def mirror_obs_batch(states: np.ndarray) -> np.ndarray:
    """mirror_obs over an [N, STATE_DIM] array in three numpy ops (the per-row
    version dominates run_update at scale)."""
    out = states.copy()
    out[:, C.MIRROR_NEGATE] *= -1
    for a, b in C.MIRROR_SWAPS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def mirror_action_batch(actions: np.ndarray) -> np.ndarray:
    """mirror_action over an [N] int array."""
    lr = actions // 6
    return np.where(lr == 1, 2, np.where(lr == 2, 1, 0)) * 6 + actions % 6


class LagEnv:
    def __init__(self, sim: BonkSim):
        self.sim = sim
        self.lag = C.INPUT_LAG
        self.max_steps = C.MAX_EPISODE_STEPS
        self.state_dim = C.STATE_DIM
        # Per seat: list of (decided_tick, action); pruned as entries expire.
        self.decisions = [[], []]
        self.episode_steps = 0
        # Heavy-meter memory, per player (not per observer — it's derived from
        # the applied keys, which both seats can see, so one tracker serves both).
        self.heavy_seen = [1.0, 1.0]        # last observed meter, HEAVY_SCALE'd
        self.heavy_seen_ticks = [0, 0]      # ticks since that observation

    def reset(self, randomize_spawns=True):
        self.sim.reset(randomize_spawns)
        self.decisions = [[], []]
        self.episode_steps = 0
        # Heavy resets to full at spawn and everyone knows it.
        self.heavy_seen = [1.0, 1.0]
        self.heavy_seen_ticks = [0, 0]
        # Per-episode lag randomization: the agent can't observe the draw, so it
        # must learn timing robust to the whole range.
        if C.INPUT_LAG_RANDOM:
            self.lag = random.randint(C.INPUT_LAG_MIN, C.INPUT_LAG_MAX)

    def set_decision(self, seat: int, action: int):
        self.decisions[seat].append((self.episode_steps, action))

    def _applied_index(self, seat: int) -> int:
        """Latest decision old enough to have taken effect (t <= now - lag)."""
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

        # Heavy tracker: while the applied heavy key is held the meter is
        # observable — record it; otherwise it regenerates unseen — count ticks.
        for p, k in ((0, k0), (1, k1)):
            if k["heavy"]:
                self.heavy_seen[p] = self.sim.players[p].heavy_power * C.HEAVY_SCALE
                self.heavy_seen_ticks[p] = 0
            else:
                self.heavy_seen_ticks[p] += 1

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

        def block(off, pi, player, applied_idx):
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
            out[off + 10] = self.heavy_seen[pi]
            out[off + 11] = min(1.0, self.heavy_seen_ticks[pi]
                                / C.HEAVY_SEEN_TICKS_NORM)

        block(0, seat, s, self._applied_index(seat))
        block(12, 1 - seat, o, self._applied_index(1 - seat))

        sx, sy = s.pos
        ox, oy = o.pos
        svx, svy = s.vel
        ovx, ovy = o.vel
        out[24] = (ox - sx) * C.POS_SCALE
        out[25] = (oy - sy) * C.POS_SCALE
        out[26] = (ovx - svx) * C.VEL_SCALE
        out[27] = (ovy - svy) * C.VEL_SCALE

        pk = index_to_keys(self._last_decided(seat))
        out[28] = 1.0 if pk["up"] else 0.0
        out[29] = 1.0 if pk["down"] else 0.0
        out[30] = 1.0 if pk["left"] else 0.0
        out[31] = 1.0 if pk["right"] else 0.0
        out[32] = 1.0 if pk["heavy"] else 0.0

        out[33] = min(1.0, self.episode_steps / self.max_steps)
        return out
