"""LagEnv: the decision-level environment — netcode, observations, terminal outcomes.

Two independent lags model the real game, both re-randomized per episode:

* **self_lag** — your own keypress -> applied. Rollback applies your input
  without waiting for the server, but not instantly: browser event -> frame
  quantization -> the deploy script's own decision cadence. Training this at
  zero is what makes contact/charge timing land late for real.
* **view_lag** — how stale your view of the OPPONENT is (the rollback
  prediction window). Their last `view_lag` ticks of input haven't arrived.

Under ``rollback`` netcode (what bonk.io actually runs) both seats' inputs
apply instantly to the authoritative sim, but each seat observes the opponent
through a client-side prediction: their state `lag` ticks ago replayed with
held inputs. The prediction is right while the opponent holds their keys and
wrong exactly when they change them — jukes live inside that window. This is
what the deploy script reads from the live game, so it is the sim-to-real
bridge; do not remove it.

**Heavy-meter memory.** The raw heavyValue is only observable while the heavy
key is held (masked to 0 otherwise), but the meter silently regenerates while
released. Without memory the observation loses information: a player who
released at 30% and re-pressed 100 ticks later looks identical to one who
released at 80%. Each player block therefore carries the last heavy value seen
while the key was held, and how long ago that was — both derived purely from
what is observable, so the observation stays map-agnostic and works
identically in the browser.

Differences from `ppo7/env.py`, all deliberate:

* single-frame observations only (the TCN window and its frame history are
  gone with the conv encoder);
* `snapshot_state`/`restore_state` are gone with tree search;
* the outcome CLASS is the primary terminal channel, using the corrected
  `Outcome` ordering (see `actions.Outcome`);
* config arrives by injection, and every scalar read inside a per-tick loop is
  hoisted onto `self` in `__init__` — `_raw_frame` alone runs twice per
  decision per env at ~50k steps/s, so a nested `self.cfg.obs.pos_scale`
  lookup in that path is a real throughput regression.
"""
from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np

from bonk import config as PC              # physics constants (gravity, heavy rates)
from bonk3.simadapter import EngineSim

from .actions import IDLE, Outcome, index_to_keys
from .config import EnvConfig
from .layout import ObsLayout
from .obs import OPP_OFF, PEND_OFF, REL_OFF, SELF_OFF, fourier_expand
from .spawns import SpawnPool


class StepResult(NamedTuple):
    """One tick's outcome.

    Attributes:
        done:    the episode ended on this tick.
        dead:    ``(seat0_dead, seat1_dead)``.
        timeout: the episode ended by draw-hazard or the safety cap.
        outcome: ``(seat0, seat1)`` `Outcome`; ``NONE`` while running.
        rewards: ``(seat0, seat1)`` terminal values — provided for standalone
                 use. A trainer should prefer `outcome`, deriving its own
                 values from the class, so reward values and critic atoms
                 cannot drift apart.
    """

    done: bool
    dead: tuple[bool, bool]
    timeout: bool
    outcome: tuple[Outcome, Outcome]
    rewards: tuple[float, float]


class LagEnv:
    """One 1v1 episode with netcode simulation, over a `bonk3` engine sim."""

    def __init__(self, sim: EngineSim, cfg: EnvConfig, layout: ObsLayout) -> None:
        self.sim = sim
        self.cfg = cfg
        self.layout = layout
        self.state_dim = layout.state_dim

        # ── hoisted hot-loop scalars (see the module docstring) ─────────────
        obs, lag, ep, spawn = cfg.obs, cfg.lag, cfg.episode, cfg.spawn
        self._raw_dim = obs.raw_frame_dim
        self._pos_scale = obs.pos_scale
        self._vel_scale = obs.vel_scale
        self._heavy_scale = obs.heavy_scale
        self._heavy_ticks_norm = obs.heavy_seen_ticks_norm
        self._fbase = layout.fourier_base
        self._ffreqs = layout.fourier_freqs
        self._ffeats = layout.fourier_feats
        self._rollback = lag.netcode == "rollback"
        self._lag_random = lag.randomize
        self._self_lag_min, self._self_lag_max = lag.self_lag_min, lag.self_lag_max
        self._view_lag_min, self._view_lag_max = lag.view_lag_min, lag.view_lag_max
        self._action_repeat = ep.action_repeat
        self._timeout_hazard = ep.timeout_hazard
        self._real_engine = cfg.engine.engine == "real"
        self._curriculum_close_m = spawn.curriculum_start_m
        self._curriculum_jitter_m = spawn.curriculum_jitter_m
        self._dt = PC.DT
        self._gravity = PC.GRAVITY
        self._heavy_drain = PC.HEAVY_DRAIN
        self._heavy_regen = PC.HEAVY_REGEN
        self._heavy_max = PC.HEAVY_MAX

        self.self_lag = lag.self_lag
        self.view_lag = lag.view_lag
        self.max_steps = ep.max_episode_steps

        self._pool: SpawnPool | None = None
        if spawn.use_pool and self._real_engine:
            assert spawn.pool_file is not None, "call EnvConfig.resolve()"
            self._pool = SpawnPool(spawn.pool_file, spawn.pool_min_sep)

        # Per seat: list of (decided_tick, action); pruned as entries expire.
        self.decisions: tuple[list[tuple[int, int]], list[tuple[int, int]]] = ([], [])
        self.episode_steps = 0
        # Heavy-meter memory, per PLAYER (not per observer — it derives from
        # the applied keys, which both seats can see, so one tracker serves both).
        self.heavy_seen = [1.0, 1.0]        # last observed meter, heavy_scale'd
        self.heavy_seen_ticks = [0, 0]      # ticks since that observation
        # Rollback: per-player state history so the opponent view can run
        # `view_lag` ticks behind.
        self.history = (deque(maxlen=self._view_lag_max + 1),
                        deque(maxlen=self._view_lag_max + 1))

    # ── observation ────────────────────────────────────────────────────────
    def _effective_self_lag(self) -> int:
        """Own input delay. Under ``delay`` the opponent view has no prediction,
        so `view_lag` is folded into the input delay instead."""
        return self.self_lag if self._rollback else self.view_lag

    def decision_state(self, seat: int) -> np.ndarray:
        """One seat's observation: ``[state_dim]`` float32.

        Layout is ``[ raw frame || Fourier(raw frame) ]``. Temporal memory is
        NOT in the observation — it lives in the learner's recurrent hidden
        state, carried across decisions.
        """
        cur = self._raw_frame(seat)
        return np.concatenate([cur, fourier_expand(cur, self.layout)])

    def _fourier(self, raw: np.ndarray) -> np.ndarray:
        """Inlined `fourier_expand` against hoisted arrays (hot path)."""
        ang = np.outer(raw[self._fbase], self._ffreqs)
        blk = np.empty((len(self._fbase), self._ffeats), dtype=np.float32)
        blk[:, 0::2] = np.sin(ang)
        blk[:, 1::2] = np.cos(ang)
        return blk.reshape(-1)

    def _raw_frame(self, seat: int) -> np.ndarray:
        """``[raw_frame_dim]`` float32 — the layout everything else indexes into."""
        s = self.sim.players[seat]
        out = np.zeros(self._raw_dim, dtype=np.float32)
        pos_scale, vel_scale = self._pos_scale, self._vel_scale
        heavy_scale, ticks_norm = self._heavy_scale, self._heavy_ticks_norm

        def block(off: int, x: float, y: float, vx: float, vy: float,
                  applied_idx: int, heavy_power: float,
                  hseen: float, hticks: float) -> None:
            k = index_to_keys(applied_idx)
            out[off] = x * pos_scale
            out[off + 1] = y * pos_scale
            out[off + 2] = vx * vel_scale
            out[off + 3] = vy * vel_scale
            out[off + 4] = heavy_power * heavy_scale if k["heavy"] else 0.0
            out[off + 5] = 1.0 if k["up"] else 0.0
            out[off + 6] = 1.0 if k["down"] else 0.0
            out[off + 7] = 1.0 if k["left"] else 0.0
            out[off + 8] = 1.0 if k["right"] else 0.0
            out[off + 9] = 1.0 if k["heavy"] else 0.0
            out[off + 10] = hseen
            out[off + 11] = min(1.0, hticks / ticks_norm)

        # Self: always the true current state (your own client is exact).
        sx, sy = s.pos
        svx, svy = s.vel
        block(SELF_OFF, sx, sy, svx, svy,
              self._applied_index(seat, self._effective_self_lag()),
              s.heavy_power, self.heavy_seen[seat], self.heavy_seen_ticks[seat])

        # Opponent: predicted under rollback, true-but-key-lagged under delay.
        if self._rollback:
            ox, oy, ovx, ovy, oaidx, ohp, ohseen, ohticks = self._opp_view(1 - seat)
        else:
            o = self.sim.players[1 - seat]
            ox, oy = o.pos
            ovx, ovy = o.vel
            oaidx = self._applied_index(1 - seat, self.view_lag)
            ohp = o.heavy_power
            ohseen = self.heavy_seen[1 - seat]
            ohticks = self.heavy_seen_ticks[1 - seat]
        block(OPP_OFF, ox, oy, ovx, ovy, oaidx, ohp, ohseen, ohticks)

        out[REL_OFF] = (ox - sx) * pos_scale
        out[REL_OFF + 1] = (oy - sy) * pos_scale
        out[REL_OFF + 2] = (ovx - svx) * vel_scale
        out[REL_OFF + 3] = (ovy - svy) * vel_scale

        pk = index_to_keys(self._last_decided(seat))
        out[PEND_OFF] = 1.0 if pk["up"] else 0.0
        out[PEND_OFF + 1] = 1.0 if pk["down"] else 0.0
        out[PEND_OFF + 2] = 1.0 if pk["left"] else 0.0
        out[PEND_OFF + 3] = 1.0 if pk["right"] else 0.0
        out[PEND_OFF + 4] = 1.0 if pk["heavy"] else 0.0
        return out

    def _opp_view(self, p: int):
        """The rollback client's rendered view of player `p`.

        Their snapshot from `view_lag` ticks ago, replayed forward with held
        inputs: ballistic while airborne, constant-vx while grounded.
        Collisions inside the window are ignored — a short window, and the same
        class of error the real client's predictor makes.

        Returns ``(x, y, vx, vy, applied_idx, heavy_power, heavy_seen,
        heavy_ticks)``.
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
        # runs only inside genuine mispredictions (keys changed in-window),
        # where the real client's prediction is wrong by definition too.
        if all(hist[-1 - j][4] == aidx for j in range(L)):
            cx, cy, cvx, cvy, _, chp, chseen, chticks, _ = hist[-1]
            return cx, cy, cvx, cvy, aidx, chp, chseen, chticks
        if grounded:
            x += vx * self._dt * L
        else:
            for _ in range(L):
                vy += self._gravity * self._dt
                x += vx * self._dt
                y += vy * self._dt
        # The prediction also runs the heavy meter forward with the held key.
        if index_to_keys(aidx)["heavy"]:
            hp = max(0.0, hp - self._heavy_drain * L)
            hseen, hticks = hp * self._heavy_scale, 0
        else:
            hp = min(self._heavy_max, hp + self._heavy_regen * L)
            hticks += L
        return x, y, vx, vy, aidx, hp, hseen, hticks

    # ── episode lifecycle ──────────────────────────────────────────────────
    def _reset_bookkeeping(self, heavy_seen: list[float] | None = None,
                           heavy_seen_ticks: list[int] | None = None) -> None:
        """Everything about a fresh episode that is NOT "where the discs are":
        the decision log, episode clock, heavy-meter memory, view history, and
        a fresh lag draw. Shared by `reset()` and `reset_from_archive()` so the
        two can never silently drift apart on what "a new episode" means.

        `heavy_seen`/`heavy_seen_ticks` let a caller restore the OBSERVED
        heavy-meter memory of a state it is replaying (see
        `reset_from_archive`) instead of the fresh-spawn default of "full,
        just seen".
        """
        self.decisions = ([], [])
        self.episode_steps = 0
        self.heavy_seen = list(heavy_seen) if heavy_seen is not None else [1.0, 1.0]
        self.heavy_seen_ticks = (list(heavy_seen_ticks) if heavy_seen_ticks is not None
                                 else [0, 0])
        # Seed history with the CURRENT (post-physics-reset) state: at t=0 the
        # clients are synced, and early ticks lag as far as real time allows
        # (t-lag clamps to spawn/restart).
        for p in (0, 1):
            self.history[p].clear()
            self.history[p].append(self._snapshot(p, IDLE))
        # Per-episode lag randomization. Uses the stdlib `random` stream, seeded
        # per worker — matching ppo7 exactly. Do not swap this for a numpy
        # Generator: it shares a stream with the draw hazard below, and
        # perturbing that changes mean episode length, which silently changes
        # what every step-based schedule means.
        if self._lag_random:
            self.self_lag = random.randint(self._self_lag_min, self._self_lag_max)
            self.view_lag = random.randint(self._view_lag_min, self._view_lag_max)

    def reset(self, randomize_spawns: bool = True, spawn_frac: float = 1.0) -> None:
        """Start a new episode from the NORMAL spawn distribution.

        `spawn_frac` < 1 asks the sim to start the two discs closer together
        (see `spawns.SpawnCurriculum`). Only the real engine supports it — the
        legacy sim has no way to place discs off-spawn.
        """
        if self._pool is not None:
            # Both discs on pre-probed survivable ground.
            self.sim.reset(randomize_spawns, positions=self._pool.sample_pair())
        elif spawn_frac < 1.0 and self._real_engine:
            self.sim.reset(randomize_spawns, spawn_frac=spawn_frac,
                           close_m=self._curriculum_close_m,
                           jitter_m=self._curriculum_jitter_m)
        else:
            self.sim.reset(randomize_spawns)
        self._reset_bookkeeping()

    def snapshot_physics(self) -> dict:
        """A restartable physics snapshot: sim state + this env's own heavy-
        meter memory (NOT lag/episode bookkeeping — a restart from this always
        draws those fresh, exactly like a normal reset; see
        `reset_from_archive`). Used by the exploiter value-swing archive
        (`ppo9.archive`) to capture candidate restart states; kept here rather
        than in that module so there is exactly one place that knows how to
        round-trip a `LagEnv`'s restartable state.
        """
        return {"phys": self.sim.get_full_state(),
                "heavy_seen": list(self.heavy_seen),
                "heavy_seen_ticks": list(self.heavy_seen_ticks)}

    def reset_from_archive(self, snapshot: dict) -> None:
        """Start a NEW episode from a previously captured `snapshot_physics()`
        blob, instead of the normal spawn distribution.

        This is a FRESH episode that happens to start from an interesting
        physics configuration — not a continuation of the episode the
        snapshot came from. Only positions/velocities/heavy-meter STATE
        carry over (via `_reset_bookkeeping`'s optional heavy_seen args);
        `episode_steps`, the decision log, and lag are reinitialized exactly
        like `reset()` does, so this episode gets the SAME full timeout
        budget and a freshly-drawn lag, matching every other episode. This is
        also why the archived physics blob's own `tick_count`/`dead` fields
        (see `EngineSim.get_full_state`) are deliberately NOT restored: they
        describe the SOURCE episode's clock, which has no bearing on this
        one's.
        """
        self.sim.set_full_state(snapshot["phys"])
        self._reset_bookkeeping(snapshot.get("heavy_seen"),
                                snapshot.get("heavy_seen_ticks"))

    def _snapshot(self, p: int, applied_idx: int):
        pl = self.sim.players[p]
        return (pl.pos[0], pl.pos[1], pl.vel[0], pl.vel[1], applied_idx,
                pl.heavy_power, self.heavy_seen[p], self.heavy_seen_ticks[p],
                pl.is_grounded())

    # ── control ────────────────────────────────────────────────────────────
    def set_decision(self, seat: int, action: int) -> None:
        """Record a seat's decision at the current tick; it applies after lag."""
        self.decisions[seat].append((self.episode_steps, action))

    def _applied_index(self, seat: int, lag: int) -> int:
        """Latest decision old enough to have taken effect (``t <= now - lag``)."""
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
        if keep_from > 0:
            del log[:keep_from]     # can never win again
        return applied

    def _last_decided(self, seat: int) -> int:
        log = self.decisions[seat]
        return log[-1][1] if log else IDLE

    def applied_actions(self) -> list[int]:
        """``[seat0, seat1]`` action indices actually in effect this tick."""
        lag = self._effective_self_lag()
        return [self._applied_index(0, lag), self._applied_index(1, lag)]

    def tick(self) -> StepResult:
        """Advance one physics tick and report terminal state."""
        a0, a1 = self.applied_actions()
        k0 = index_to_keys(a0)
        k1 = index_to_keys(a1)
        self.sim.step(k0, k1)
        self.episode_steps += 1

        # Heavy tracker: while the applied heavy key is held the meter is
        # observable — record it; otherwise it regenerates unseen — count ticks.
        # Must run BEFORE the history append, so snapshots carry the update.
        for p, k in ((0, k0), (1, k1)):
            if k["heavy"]:
                self.heavy_seen[p] = self.sim.players[p].heavy_power * self._heavy_scale
                self.heavy_seen_ticks[p] = 0
            else:
                self.heavy_seen_ticks[p] += 1

        if self._rollback:
            self.history[0].append(self._snapshot(0, a0))
            self.history[1].append(self._snapshot(1, a1))

        dead = (self.sim.is_dead(0), self.sim.is_dead(1))
        # Stochastic draw: a constant per-DECISION hazard rather than a
        # deadline, so the hazard means what the config says regardless of
        # action_repeat. max_steps survives only as a safety cap bounding the
        # geometric tail. Feeding elapsed time to the agent instead made the
        # problem non-stationary and it stalled while the clock was low.
        rolled = False
        if self._timeout_hazard > 0.0 and self.episode_steps % self._action_repeat == 0:
            rolled = random.random() < self._timeout_hazard
        timeout = not any(dead) and (rolled or self.episode_steps >= self.max_steps)
        done = any(dead) or timeout

        rewards = (0.0, 0.0)
        outcome = (Outcome.NONE, Outcome.NONE)
        if done:
            rv = self.cfg.rewards
            if timeout or (dead[0] and dead[1]):
                rewards = (rv.draw, rv.draw)
                outcome = (Outcome.DRAW, Outcome.DRAW)
            elif dead[1]:
                rewards = (rv.win, rv.loss)
                outcome = (Outcome.WIN, Outcome.LOSS)
            else:
                rewards = (rv.loss, rv.win)
                outcome = (Outcome.LOSS, Outcome.WIN)
        return StepResult(done=done, dead=dead, timeout=timeout,
                          outcome=outcome, rewards=rewards)
