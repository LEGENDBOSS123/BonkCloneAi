"""bonk2 LagEnv: decision-level env with netcode simulation, joint discrete
actions, normalized 34-dim observations, and sparse terminal rewards.

Two independent lags model the real game (both randomized per episode):
  SELF_LAG — your own keypress -> applied. Rollback applies your input without
    waiting for the server, but not instantly: browser event -> frame
    quantization -> the deploy script's own decision cadence. Training this at
    zero is what makes contact/charge timing (heavy) land late for real.
  VIEW_LAG — how stale your view of the OPPONENT is (the rollback prediction
    window). Their last VIEW_LAG ticks of input haven't arrived.

Netcode (config.NETCODE) — the sim-to-real bridge:
- "rollback" (what bonk.io actually runs): both seats' inputs apply INSTANTLY
  to the authoritative sim, but each seat OBSERVES the opponent through a
  client-side prediction — the opponent's state `lag` ticks ago replayed with
  held inputs (ballistic extrapolation here; collisions ignored over the short
  window). The prediction is right while the opponent holds their keys and
  wrong exactly when they change them — jukes live inside this window. This
  matches what play3.mjs reads from the live game (rendered predicted position
  + last-received keys).
- "delay": legacy v1 model — a decision made at tick t governs physics from
  tick t + lag for BOTH seats, opponent observed perfectly.

`lag` is randomized per episode (network jitter) in both modes.

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
from collections import deque

import numpy as np

from bonk import config as PC          # physics constants (gravity, heavy rates)
from bonk.sim import BonkSim, index_to_keys

from . import config as C

IDLE = 0

# Outcome classes — the critic's 3-way target. OUT_NONE marks a non-terminal
# tick (no outcome yet) and is never a label.
OUT_WIN, OUT_LOSS, OUT_DRAW, OUT_NONE = 0, 1, 2, -1


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


def mirror_obs_into(dst: np.ndarray, src: np.ndarray) -> None:
    """mirror_obs_batch writing into a PREALLOCATED dst, so the update does not
    allocate a fresh [N, 34] every time. dst may alias src."""
    if dst is not src:
        np.copyto(dst, src)
    dst[:, C.MIRROR_NEGATE] *= -1
    for a, b in C.MIRROR_SWAPS:
        dst[:, [a, b]] = dst[:, [b, a]]


def mirror_action_into(dst: np.ndarray, src: np.ndarray) -> None:
    lr = src // 6
    np.copyto(dst, np.where(lr == 1, 2, np.where(lr == 2, 1, 0)) * 6 + src % 6)


def mirror_action_batch(actions: np.ndarray) -> np.ndarray:
    """mirror_action over an [N] int array."""
    lr = actions // 6
    return np.where(lr == 1, 2, np.where(lr == 2, 1, 0)) * 6 + actions % 6


class LagEnv:
    def __init__(self, sim: BonkSim):
        self.sim = sim
        self.self_lag = C.SELF_LAG    # own keypress -> applied (both seats)
        self.view_lag = C.VIEW_LAG    # staleness of your view of the opponent
        self.max_steps = C.MAX_EPISODE_STEPS
        self.state_dim = C.STATE_DIM
        # Per seat: list of (decided_tick, action); pruned as entries expire.
        self.decisions = [[], []]
        self.episode_steps = 0
        # Heavy-meter memory, per player (not per observer — it's derived from
        # the applied keys, which both seats can see, so one tracker serves both).
        self.heavy_seen = [1.0, 1.0]        # last observed meter, HEAVY_SCALE'd
        self.heavy_seen_ticks = [0, 0]      # ticks since that observation
        # Rollback: per-player state history so the opponent view can run
        # `lag` ticks behind. Snapshot per tick:
        # (x, y, vx, vy, applied_idx, heavy_power, heavy_seen, heavy_ticks, grounded)
        self.history = (deque(maxlen=C.VIEW_LAG_MAX + 1),
                        deque(maxlen=C.VIEW_LAG_MAX + 1))

    def _self_lag(self) -> int:
        """Own input delay. Under "delay" (legacy v1) the opponent view has no
        prediction, so VIEW_LAG is folded into the input delay instead."""
        return self.self_lag if C.NETCODE == "rollback" else self.view_lag

    def reset(self, randomize_spawns=True, spawn_frac=1.0):
        # spawn_frac < 1 asks the sim to sometimes start the two discs close
        # together (config.SPAWN_CURRICULUM). Only the real engine supports it;
        # the legacy Box2D sim has no way to place discs off-spawn.
        if spawn_frac < 1.0 and C.ENGINE == "real":
            self.sim.reset(randomize_spawns, spawn_frac=spawn_frac,
                           close_m=C.SPAWN_CURRICULUM_START_M,
                           jitter_m=C.SPAWN_CURRICULUM_JITTER_M)
        else:
            self.sim.reset(randomize_spawns)
        self.decisions = [[], []]
        self.episode_steps = 0
        # Heavy resets to full at spawn and everyone knows it.
        self.heavy_seen = [1.0, 1.0]
        self.heavy_seen_ticks = [0, 0]
        # Seed the history with the spawn state: at t=0 the clients are synced,
        # and early ticks lag as far as real time allows (t-lag clamps to spawn).
        for p in (0, 1):
            self.history[p].clear()
            self.history[p].append(self._snapshot(p, IDLE))
        # Per-episode lag randomization: the agent can't observe the draw, so it
        # must learn timing robust to the whole range.
        if C.LAG_RANDOM:
            self.self_lag = random.randint(C.SELF_LAG_MIN, C.SELF_LAG_MAX)
            self.view_lag = random.randint(C.VIEW_LAG_MIN, C.VIEW_LAG_MAX)

    def _win_reward(self) -> float:
        """WIN_REWARD scaled from 1.0 down to (1 - WIN_TIME_DECAY) across the
        clock. Stays Markov: elapsed time is already an observed feature
        (obs[33]), and it is still a single terminal reward."""
        if C.WIN_TIME_DECAY <= 0.0:
            return C.WIN_REWARD
        frac = min(1.0, self.episode_steps / self.max_steps)
        return C.WIN_REWARD * (1.0 - C.WIN_TIME_DECAY * frac)

    def _snapshot(self, p: int, applied_idx: int):
        pl = self.sim.players[p]
        return (pl.pos[0], pl.pos[1], pl.vel[0], pl.vel[1], applied_idx,
                pl.heavy_power, self.heavy_seen[p], self.heavy_seen_ticks[p],
                pl.is_grounded())

    def set_decision(self, seat: int, action: int):
        self.decisions[seat].append((self.episode_steps, action))

    def _applied_index(self, seat: int, lag: int) -> int:
        """Latest decision old enough to have taken effect (t <= now - lag)."""
        cutoff = self.episode_steps - lag
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
        lag = self._self_lag()
        return [self._applied_index(0, lag), self._applied_index(1, lag)]

    def tick(self):
        a0, a1 = self.applied_actions()
        k0 = index_to_keys(a0)
        k1 = index_to_keys(a1)
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

        if C.NETCODE == "rollback":
            self.history[0].append(self._snapshot(0, a0))
            self.history[1].append(self._snapshot(1, a1))

        dead = [self.sim.is_dead(0), self.sim.is_dead(1)]
        timeout = not any(dead) and self.episode_steps >= self.max_steps
        done = any(dead) or timeout

        # ppo4: the return is EXACTLY one of three terminal values, nothing
        # else. No per-step cost and no time decay — the critic is a 3-way
        # classifier over {win, loss, draw}, so any continuous component would
        # make the classification target ill-posed (see config.OUTCOMES).
        rewards = [0.0, 0.0]
        outcome = [OUT_NONE, OUT_NONE]
        if done:
            if timeout or (dead[0] and dead[1]):
                rewards = [C.DRAW_REWARD, C.DRAW_REWARD]
                outcome = [OUT_DRAW, OUT_DRAW]
            elif dead[1]:
                rewards = [self._win_reward(), C.LOSS_REWARD]
                outcome = [OUT_WIN, OUT_LOSS]
            else:
                rewards = [C.LOSS_REWARD, self._win_reward()]
                outcome = [OUT_LOSS, OUT_WIN]
        return {"rewards": rewards, "done": done, "dead": dead,
                "timeout": timeout, "outcome": outcome}

    def _opp_view(self, p: int):
        """The rollback client's rendered view of player p: their snapshot from
        `lag` ticks ago replayed forward with held inputs. Ballistic while
        airborne, constant-vx while grounded; collisions inside the window are
        ignored (short window, same class of error the real predictor makes).
        Returns (x, y, vx, vy, applied_idx, heavy_power, heavy_seen, heavy_ticks).
        """
        hist = self.history[p]
        # Snapshot at t-L (hist[-1] is t), clamped to spawn early in the round.
        L = min(self.view_lag, len(hist) - 1)
        x, y, vx, vy, aidx, hp, hseen, hticks, grounded = hist[-1 - L]
        if L == 0:
            return x, y, vx, vy, aidx, hp, hseen, hticks
        # Exact path: if the opponent's applied keys were constant across the
        # whole window, the client's held-input resim reproduces reality
        # tick-for-tick (collisions included) — so the true CURRENT state IS
        # the prediction. This is the common case; the ballistic guess below
        # only runs inside genuine mispredictions (keys changed in-window),
        # where the real client's prediction is wrong by definition too.
        if all(hist[-1 - j][4] == aidx for j in range(L)):
            cx, cy, cvx, cvy, _, chp, chseen, chticks, _ = hist[-1]
            return cx, cy, cvx, cvy, aidx, chp, chseen, chticks
        if grounded:
            x += vx * PC.DT * L
        else:
            for _ in range(L):
                vy += PC.GRAVITY * PC.DT
                x += vx * PC.DT
                y += vy * PC.DT
        # The prediction also runs the heavy meter forward with the held key.
        if index_to_keys(aidx)["heavy"]:
            hp = max(0.0, hp - PC.HEAVY_DRAIN * L)
            hseen, hticks = hp * C.HEAVY_SCALE, 0
        else:
            hp = min(PC.HEAVY_MAX, hp + PC.HEAVY_REGEN * L)
            hticks += L
        return x, y, vx, vy, aidx, hp, hseen, hticks

    def decision_state(self, seat: int) -> np.ndarray:
        s = self.sim.players[seat]
        out = np.zeros(C.STATE_DIM, dtype=np.float32)

        def block(off, x, y, vx, vy, applied_idx, heavy_power, hseen, hticks):
            k = index_to_keys(applied_idx)
            out[off] = x * C.POS_SCALE
            out[off + 1] = y * C.POS_SCALE
            out[off + 2] = vx * C.VEL_SCALE
            out[off + 3] = vy * C.VEL_SCALE
            out[off + 4] = heavy_power * C.HEAVY_SCALE if k["heavy"] else 0.0
            out[off + 5] = 1.0 if k["up"] else 0.0
            out[off + 6] = 1.0 if k["down"] else 0.0
            out[off + 7] = 1.0 if k["left"] else 0.0
            out[off + 8] = 1.0 if k["right"] else 0.0
            out[off + 9] = 1.0 if k["heavy"] else 0.0
            out[off + 10] = hseen
            out[off + 11] = min(1.0, hticks / C.HEAVY_SEEN_TICKS_NORM)

        # Self: always the true current state (your own client is exact).
        sx, sy = s.pos
        svx, svy = s.vel
        block(0, sx, sy, svx, svy, self._applied_index(seat, self._self_lag()),
              s.heavy_power, self.heavy_seen[seat], self.heavy_seen_ticks[seat])

        # Opponent: predicted view under rollback, true-but-key-lagged under delay.
        if C.NETCODE == "rollback":
            ox, oy, ovx, ovy, oaidx, ohp, ohseen, ohticks = self._opp_view(1 - seat)
        else:
            o = self.sim.players[1 - seat]
            ox, oy = o.pos
            ovx, ovy = o.vel
            oaidx = self._applied_index(1 - seat, self.view_lag)
            ohp = o.heavy_power
            ohseen = self.heavy_seen[1 - seat]
            ohticks = self.heavy_seen_ticks[1 - seat]
        block(12, ox, oy, ovx, ovy, oaidx, ohp, ohseen, ohticks)

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
