"""LagEnv: the decision-level environment — netcode simulation, joint discrete
actions, normalized observations, and sparse terminal rewards.

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
  matches what the deploy script reads from the live game (rendered predicted
  position + last-received keys).
- "delay": legacy v1 model — a decision made at tick t governs physics from
  tick t + lag for BOTH seats, opponent observed perfectly.

`lag` is randomized per episode (network jitter) in both modes.

Heavy-meter memory. The raw heavyValue is only observable while the heavy key is
held (masked to 0 otherwise), but the meter silently regenerates at 5/tick while
released. Without memory the obs loses information: a player who released at 30%
and re-pressed 100 ticks later looks exactly like one who released at 80%. So
each player block carries the last heavy value seen while the key was held and
how many ticks ago that was (normalized by the 200-tick full-regen span, capped
at 1) — both derived purely from what is observable, so the obs stays
map-agnostic and works identically in the browser.

RAW FRAME layout (config.RAW_FRAME_DIM = 33), before Fourier expansion:
  [ 0-11] self : x, y, vx, vy, heavyValue(masked), up, down, left, right,
                 heavyKey, lastHeavySeen, ticksSinceSeen/200 (cap 1)
  [12-23] opp  : same 12
  [24-27] rel  : dx, dy, dvx, dvy
  [28-32] pend : own last DECIDED action bits [up, down, left, right, heavy]
There is deliberately NO draw-clock feature: feeding elapsed time made the
problem non-stationary and the agent exploited exactly that, stalling while the
clock was low (see config.TIMEOUT_HAZARD, which replaced the deadline with a
per-decision hazard).

What `decision_state` returns on top of that raw frame depends on the temporal
encoder (config.RECURRENT): a single frame + its Fourier block for the minGRU
(memory lives in the hidden state), or a TCN_WINDOW-frame raw window + one
Fourier block for the legacy conv encoder.
"""

import random
from collections import deque

import numpy as np

from bonk import config as PC          # physics constants (gravity, heavy rates)
from bonk.sim import index_to_keys
from bonk3.simadapter import EngineSim

from . import config as C
# Mirror augmentation lives in its own module; re-exported here because
# the obs layout it transforms is defined by this file.
from .mirror import (mirror_action, mirror_action_batch,  # noqa: F401
                     mirror_obs, mirror_obs_batch)

# NeRF-style Fourier frequencies pi*2^k, precomputed once (see config.FOURIER_*).
_FOURIER_FREQS = (np.pi * (2.0 ** (C.FOURIER_K_START + np.arange(C.FOURIER_L)))).astype(np.float32)
_FOURIER_BASE = np.asarray(C.FOURIER_BASE_DIMS, dtype=np.intp)

IDLE = 0

# ── survivable-spawn pool ───────────────────────────────────────────────────
# Positions pre-probed by ppo6.gen_spawns: every point is stable ground, so
# neither disc ever starts in a death trap, and "self" is spread across the
# whole map (which validates both seats + mirror augmentation). Loaded once per
# process, lazily, so workers that don't use it pay nothing.
_SPAWN_POOL = None


def _spawn_pool():
    global _SPAWN_POOL
    if _SPAWN_POOL is None:
        import json
        with open(C.SPAWN_POOL_FILE) as f:
            _SPAWN_POOL = np.asarray(json.load(f)["positions"], dtype=np.float32)
    return _SPAWN_POOL


def _sample_spawn_pair():
    """Two survivable positions at least SPAWN_POOL_MIN_SEP apart (np.random is
    seeded per worker, so this inherits the worker's stream)."""
    pool = _spawn_pool()
    p0 = pool[np.random.randint(len(pool))]
    for _ in range(16):
        p1 = pool[np.random.randint(len(pool))]
        if np.hypot(*(p1 - p0)) >= C.SPAWN_POOL_MIN_SEP:
            return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))

# Outcome classes — the critic's 3-way target. OUT_NONE marks a non-terminal
# tick (no outcome yet) and is never a label.
OUT_WIN, OUT_LOSS, OUT_DRAW, OUT_NONE = 0, 1, 2, -1


class LagEnv:
    def __init__(self, sim: EngineSim):
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
        # Per-seat RAW-frame history feeding the temporal encoder (newest-first);
        # cleared on reset so no motion leaks across an episode boundary.
        self._fhist = (deque(maxlen=C.TCN_WINDOW),
                       deque(maxlen=C.TCN_WINDOW))

    def _self_lag(self) -> int:
        """Own input delay. Under "delay" (legacy v1) the opponent view has no
        prediction, so VIEW_LAG is folded into the input delay instead."""
        return self.self_lag if C.NETCODE == "rollback" else self.view_lag

    def decision_state(self, seat: int) -> np.ndarray:
        """One seat's STATE_DIM observation, in whichever layout the configured
        temporal encoder wants.

        RECURRENT (minGRU):  [ raw frame || Fourier(raw frame) ]           (129)
        windowed (TCN):      [ raw window x TCN_WINDOW || Fourier(current) ] (1152)

        The window is NEWEST-FIRST, which keeps `window[0:RAW_FRAME_DIM]` the
        present frame: every raw index (own position, own block, the mirror
        table's first frame) then means the same thing in both layouts, and the
        conv still reads the frames as a temporal sequence.
        """
        cur = self._raw_frame(seat)
        if C.RECURRENT:
            # Single current frame; temporal memory lives in the minGRU hidden
            # state carried across decisions (no window).
            return np.concatenate([cur, self._fourier_expand(cur)])
        h = self._fhist[seat]
        # On reset the window is primed with copies of the first frame, so a
        # fresh episode reports zero motion rather than a spurious jump from
        # whatever the previous episode ended on.
        if not h:
            for _ in range(C.TCN_WINDOW):
                h.appendleft(cur.copy())
        else:
            h.appendleft(cur)                 # maxlen=TCN_WINDOW evicts the oldest
        window = np.concatenate(list(h))      # (TCN_WINDOW * RAW_FRAME_DIM,)
        return np.concatenate([window, self._fourier_expand(cur)])

    def reset(self, randomize_spawns=True, spawn_frac=1.0):
        # spawn_frac < 1 asks the sim to sometimes start the two discs close
        # together (config.SPAWN_CURRICULUM). Only the real engine supports it;
        # the legacy Box2D sim has no way to place discs off-spawn.
        if C.USE_SPAWN_POOL and C.ENGINE == "real":
            # Both discs at pre-probed survivable ground — no death-trap spawns,
            # and "self" spread over the whole map (validates both seats).
            self.sim.reset(randomize_spawns,
                           positions=_sample_spawn_pair())
        elif spawn_frac < 1.0 and C.ENGINE == "real":
            self.sim.reset(randomize_spawns, spawn_frac=spawn_frac,
                           close_m=C.SPAWN_CURRICULUM_START_M,
                           jitter_m=C.SPAWN_CURRICULUM_JITTER_M)
        else:
            self.sim.reset(randomize_spawns)
        self.decisions = [[], []]
        self.episode_steps = 0
        for h in self._fhist:
            h.clear()          # no motion leaks across the boundary
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

    # ── fork / rewind (lookahead search) ────────────────────────────────────
    # For a caller that wants to simulate a SPECULATIVE continuation and then
    # put the real episode back exactly as it was -- e.g. lookahead.py's
    # N-rollout search. Physics goes through `EngineSim.get_full_state`/
    # `set_full_state`, NOT `engine.state_json`/`set_state_json` directly --
    # the engine's own save/restore only covers the physics, while EngineSim
    # caches a fast-path readout AND a death flag on top that a bare
    # state_json round-trip leaves stale (see that method's docstring for why
    # this bit LagEnv's own snapshot test the first time it was written).
    # Those calls are also the "slow, for debugging/parity work only" ones:
    # measured ~295us per save+restore, ~28x a plain step() (see
    # ARCHITECTURE.md's tree-search notes) -- fine for a handful of
    # candidates, not something to call per-tick or at training scale without
    # measuring again. Everything else here is a cheap Python copy.
    def snapshot_state(self):
        return {
            "phys": self.sim.get_full_state(),
            "decisions": [list(d) for d in self.decisions],
            "episode_steps": self.episode_steps,
            "heavy_seen": list(self.heavy_seen),
            "heavy_seen_ticks": list(self.heavy_seen_ticks),
            "history": tuple(deque(h, maxlen=h.maxlen) for h in self.history),
            "fhist": tuple(deque(h, maxlen=h.maxlen) for h in self._fhist),
            "self_lag": self.self_lag,
            "view_lag": self.view_lag,
        }

    def restore_state(self, snap) -> None:
        self.sim.set_full_state(snap["phys"])
        self.decisions = [list(d) for d in snap["decisions"]]
        self.episode_steps = snap["episode_steps"]
        self.heavy_seen = list(snap["heavy_seen"])
        self.heavy_seen_ticks = list(snap["heavy_seen_ticks"])
        self.history = tuple(deque(h, maxlen=h.maxlen) for h in snap["history"])
        self._fhist = tuple(deque(h, maxlen=h.maxlen) for h in snap["fhist"])
        self.self_lag = snap["self_lag"]
        self.view_lag = snap["view_lag"]

    def _win_reward(self) -> float:
        """WIN_REWARD scaled from 1.0 down to (1 - WIN_TIME_DECAY) across the
        clock, so a faster win is worth more.

        NOTE: this is only Markov if elapsed time is observable, and it is NOT —
        the draw-clock feature was removed from the obs. Kept at
        WIN_TIME_DECAY = 0.0 (i.e. disabled), which is also what makes the
        3-atom categorical critic an exact classifier. Re-enabling it means
        putting a clock feature back in the observation first.
        """
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
        # Stochastic draw: a constant per-decision hazard rather than a
        # deadline. Rolled once per DECISION (not per tick) so the hazard means
        # what the config says regardless of ACTION_REPEAT. max_steps survives
        # only as a safety cap that bounds the geometric tail.
        rolled = False
        if C.TIMEOUT_HAZARD > 0.0 and self.episode_steps % C.ACTION_REPEAT == 0:
            rolled = random.random() < C.TIMEOUT_HAZARD
        timeout = not any(dead) and (rolled or self.episode_steps >= self.max_steps)
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

    @staticmethod
    def _fourier_expand(raw: np.ndarray) -> np.ndarray:
        """FOURIER_BLOCK-dim Fourier features for one raw frame's base dims.
        Layout per base dim j: sin_0,cos_0,sin_1,cos_1,... (matches config mirror)."""
        ang = np.outer(raw[_FOURIER_BASE], _FOURIER_FREQS)   # (12, L)
        blk = np.empty((len(_FOURIER_BASE), C.FOURIER_FEATS), dtype=np.float32)
        blk[:, 0::2] = np.sin(ang)
        blk[:, 1::2] = np.cos(ang)
        return blk.reshape(-1)

    def _raw_frame(self, seat: int) -> np.ndarray:
        """One RAW_FRAME_DIM (33) observation -- the layout everything indexes
        into, BEFORE Fourier expansion. The temporal window is built from these."""
        s = self.sim.players[seat]
        out = np.zeros(C.RAW_FRAME_DIM, dtype=np.float32)

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

        return out
