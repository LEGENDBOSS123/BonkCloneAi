"""Value-swing restart archive: detection, reconstruction, and storage.

Lets an exploiter repeatedly practice states where ITS OWN critic's value
estimate swung sharply, in a temporally meaningful order:

* **recovery** — a LOW point followed later by a HIGH one (a risky spot it
  successfully converted);
* **fumble** — a HIGH point followed later by a LOW one (an advantage it let
  slip).

This changes only the INITIAL-STATE distribution of exploiter episodes. It is
never reward shaping (nothing here touches `b_r`/GAE/the outcome tables), and
it must never influence whether the exploiter is judged to be doing well —
see `Trainer._finish_episode`, which excludes archive-started episodes from
`League.record_exp_result`, the only call in the exploiter branch that feeds
a league/PFSP/Elo/promotion statistic.

The MAIN branch excludes them too, which is not redundant: an archive-started
episode still in flight when its exploiter graduates ends after the phase has
already flipped back to "main", and would otherwise reach Elo/PFSP and the
reported win rate. Note also that `done[i, DONE_ARCHIVE]` describes the episode
that just STARTED (the worker writes it at the reset), so the exclusion is
driven by `Trainer._archive_started` — the provenance of the episode being
scored — while hidden-seeding uses the `done` flag directly. Reading one for
both is off by a whole episode, in both directions.

Two correctness problems this module exists to solve, both stemming from the
same root cause — a recurrent hidden state is a function of the OBSERVATIONS
that produced it *and the network weights live when it was computed*:

1. **Critic-version consistency.** The retained per-env window can span
   several PPO updates (`archive.history` decisions vs. ~15-32 decisions
   between updates at production scale), so two stored VALUES may have come
   from different critic weights and are not directly comparable.
2. **Recurrent-state reconstruction.** A stored hidden state cannot just be
   reused under the CURRENT weights (that produces a hidden the current
   network never actually computes), and starting from `h=0` at an arbitrary
   mid-episode point plus a fixed burn-in count is *not* justified — minGRU's
   `h_t=(1-z_t)h_{t-1}+z_t h~_t` forgets a wrong initial condition at
   whatever rate `z_t` happens to run, which a constant discard count does
   not bound. See `PpoConfig.burn_in`'s OWN convention for the contrast: that
   recomputes from the REAL captured `ha0`, using burn-in only to absorb a
   few epochs of weight drift — not to invent an unknown hidden state.

**The fix used for both:** `h=0` is not an approximation everywhere — it is
EXACT at a true episode-boundary reset, because that is the same fact
`Trainer._ingest` already relies on for every normal episode start. So the
replay is anchored to such a boundary and covers only the segment after it,
re-run under the CURRENT weights via `MinGRUNet.forward_seq(h0=zeros)`. If a
window has no boundary in it (the episode has run longer than the window),
that env is simply skipped for this cycle — a coverage gap, not a guess.

That splits into two functions, and using the wrong one is a silent failure:
`replay_from_reset` does the replay on a segment the caller has ALREADY
anchored (no reset mask needed — there are no interior resets in it by
construction), while `reconstruct` first SEARCHES an arbitrary window for the
last boundary and then delegates. Handing `reconstruct` an already-sliced
segment always returns None, because the row that proves the anchor
(`done[idx-1] > 0`) sits one row before the slice and is not in it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .agent import PPOAgent
from .config import ArchiveConfig


# ── recurrent-consistent value reconstruction ────────────────────────────────
def last_reset_index(done: np.ndarray) -> int | None:
    """Index of the last TRUE episode-boundary reset inside `done[H]`.

    A row `t` is a reset point if `done[t-1] > 0` (the row before it ended an
    episode) OR `t == 0` is being treated as a boundary by the caller — this
    function only reports an INTERIOR boundary; row 0 being "the start of
    whatever we happened to retain" is NOT a verified reset and must not be
    treated as one (that is exactly the mistake this module exists to avoid).

    Returns:
        The largest `t` with `done[t-1] > 0`, or None if no such row exists.
    """
    if len(done) < 2:
        return None
    idx = np.nonzero(done[:-1] > 0)[0]
    return int(idx[-1]) + 1 if len(idx) else None


def replay_from_reset(agent: PPOAgent, obs: np.ndarray
                      ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Values + final actor/critic hidden for `obs[K,D]`, GIVEN the caller
    already guarantees row 0 is a true post-reset observation — `h=0` there
    is then exact by definition, with no search required.

    Both hidden states are reconstructed the SAME way: an actor's recurrent
    update `h_t = f(h_{t-1}, x_t)` is a pure function of the OBSERVATION
    sequence, exactly like the critic's — it does not see the action, only
    `x_t` (env obs + the FiGAR hold feature). There is nothing about the
    actor that makes zero-seeding more or less justified than for the critic;
    both get the identical reset-anchored replay.

    This is the shared core `reconstruct` (searches an arbitrary window for
    where to anchor) and archive-restart hidden seeding (already HAS an
    anchored segment — an `ArchiveEntry`'s own `context_obs`, sliced to start
    exactly at its verified reset — so searching it again would fail: the row
    that would PROVE it's a reset, `done[idx-1]>0`, was the one row before
    the slice and is not IN it) both build on.
    """
    dev = agent.device
    x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(dev)[None]
    with torch.no_grad():
        a_out, ha, _ = agent.actor.forward_seq(x)      # h0=None -> zeros; EXACT here
        c_out, hc, _ = agent.critic.forward_seq(x)
        values = agent.value_of(c_out)[0].cpu().numpy()
    return values, ha[0, -1], hc[0, -1]


def reconstruct(agent: PPOAgent, obs: np.ndarray, done: np.ndarray
                ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor] | None:
    """Recurrent-consistent (values, actor hidden, critic hidden) for the
    segment since the last verified reset, searched for WITHIN an arbitrary
    retained window that may span multiple episodes.

    Args:
        agent: whose CURRENT actor/critic weights to score with.
        obs:   ``[H, D]`` this env's retained observations, oldest first.
        done:  ``[H]`` this env's retained done flags, same order.

    Returns:
        ``(values[K], h_a_final[gru_hidden], h_c_final[gru_hidden])`` for the
        ``K = H - reset_idx`` rows from the last reset to the end of the
        window, or None if the window contains no verified reset to anchor to.
    """
    idx = last_reset_index(done)
    if idx is None:
        return None
    seg_obs = obs[idx:]
    if len(seg_obs) == 0:
        return None
    # By construction (`idx` is the LAST reset in the whole window) there is
    # no INTERIOR reset within this segment to mask for — `replay_from_reset`
    # already assumes exactly that.
    return replay_from_reset(agent, seg_obs)


# ── swing detection over a reconstructed value curve ─────────────────────────
@dataclass
class Swing:
    direction: str          # "recovery" | "fumble"
    crit_idx: int            # index of the extremum, WITHIN THE PASSED ARRAY
    now_idx: int             # index of the later confirming point
    magnitude: float


def detect_swing(values: np.ndarray, cfg: ArchiveConfig) -> Swing | None:
    """The largest valid ordered swing in a RECURRENT-CONSISTENT value curve.

    `values` must already be a single contiguous, single-episode-segment,
    single-critic-version sequence (see `reconstruct`) — this function adds no
    version or hidden-state handling of its own, only the temporal-ordering
    and terminal-exclusion logic.

    Recovery = an earlier LOW followed by a later HIGH; fumble = an earlier
    HIGH followed by a later LOW. Rows within `terminal_exclusion` of the
    array's OWN end are excluded from being either the extremum or the later
    point — this array already stops at "now" (the live edge of collection),
    so if the episode is about to end, that exclusion is applied by the caller
    only passing a window that itself excludes the trailing terminal rows (see
    `ValueTracker.scan`).

    Returns:
        The single largest-magnitude qualifying `Swing`, or None.
    """
    n = len(values)
    if n < 2:
        return None
    best: Swing | None = None

    if cfg.recovery_enabled:
        running_min = np.minimum.accumulate(values)
        # running_min[t] uses values[0..t], i.e. it already EXCLUDES values[t]
        # itself as the "later" point via the shift below.
        mag = values[1:] - running_min[:-1]
        if mag.size:
            j = int(np.argmax(mag))            # now_idx - 1 in the shifted array
            if mag[j] > (best.magnitude if best else -np.inf):
                crit = int(np.argmin(values[:j + 1]))
                if mag[j] >= cfg.value_threshold:
                    best = Swing("recovery", crit, j + 1, float(mag[j]))

    if cfg.fumble_enabled:
        running_max = np.maximum.accumulate(values)
        mag = running_max[:-1] - values[1:]
        if mag.size:
            j = int(np.argmax(mag))
            if mag[j] >= cfg.value_threshold and mag[j] > (best.magnitude if best else -np.inf):
                crit = int(np.argmax(values[:j + 1]))
                best = Swing("fumble", crit, j + 1, float(mag[j]))

    return best


# ── per-env retained window (obs + value + done), tracked subset only ───────
class ValueTracker:
    """Ring buffer of (obs, value, done) for the `archive.track_frac` subset.

    Cheap by construction: no physics, just floats, and only ever allocated
    for the fixed subset of envs that also retain physics (see
    `bonkenv.collect`) — detecting a swing in an env whose past can never be
    materialized into a restart state is pure waste (the correctness-review
    point about wasted bookkeeping).

    `value` is the RAW, live, possibly-stale-critic-version value from
    `act_step` — used ONLY as a cheap trigger to decide whether an env is
    worth the (still cheap, but non-zero) `reconstruct` call this cycle. The
    authoritative decision always runs on `reconstruct`'s recomputed curve.
    """

    def __init__(self, tracked_idx: np.ndarray, capacity: int, state_dim: int) -> None:
        self.idx = tracked_idx                  # GLOBAL env indices this tracks
        self.n = len(tracked_idx)
        self.cap = capacity
        self.obs = np.zeros((capacity, self.n, state_dim), np.float32)
        self.value = np.zeros((capacity, self.n), np.float32)
        self.done = np.zeros((capacity, self.n), np.float32)
        self.t = 0                              # next write slot (mod cap)
        self.filled = 0                         # rows written so far, capped at cap

    def push(self, obs_full: np.ndarray, value_full: np.ndarray,
             done_full: np.ndarray) -> None:
        """`obs_full[E,D]`, `value_full[E]`, `done_full[E]` — the FULL per-env
        arrays for this cycle; only the tracked columns are retained."""
        s = self.t % self.cap
        self.obs[s] = obs_full[self.idx]
        self.value[s] = value_full[self.idx]
        self.done[s] = done_full[self.idx]
        self.t += 1
        self.filled = min(self.filled + 1, self.cap)

    def window(self, local_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chronologically-ordered `(obs[L,D], value[L], done[L])` for the
        `local_i`-th tracked env, oldest first, `L = self.filled`."""
        if self.filled < self.cap:
            sl = slice(0, self.filled)
            return self.obs[sl, local_i], self.value[sl, local_i], self.done[sl, local_i]
        order = (np.arange(self.cap) + self.t) % self.cap
        return self.obs[order, local_i], self.value[order, local_i], self.done[order, local_i]

    def clear(self) -> None:
        """Called on every league phase transition (see `Trainer._reset_collection`)
        — a value from a different critic/phase must never leak into detection."""
        self.t = 0
        self.filled = 0


# ── the archive itself ───────────────────────────────────────────────────────
@dataclass
class ArchiveEntry:
    """One restart candidate. `phys`/`heavy_seen`/`heavy_seen_ticks` match
    `LagEnv.snapshot_physics()`'s format exactly (round-tripped through
    `reset_from_archive` unchanged)."""

    id: int
    phys: Any
    heavy_seen: list
    heavy_seen_ticks: list
    context_obs: np.ndarray       # [K, D] preceding window, for hidden reconstruction
    context_done: np.ndarray      # [K]
    direction: str
    magnitude: float
    source_env: int
    source_step: int
    lookback_offset: int
    x: float                      # player0 x at capture, for cheap dedup
    y: float


class Archive:
    """Bounded FIFO ring of `ArchiveEntry`, with lightweight dedup.

    IDs are monotonically increasing and NEVER reused, even as old entries are
    evicted — this is what lets a worker report back "I used entry #N" and
    have that be an unambiguous reference regardless of how much the archive
    has changed since (see `bonkenv.collect`'s request/response protocol).
    """

    def __init__(self, cfg: ArchiveConfig) -> None:
        self.cfg = cfg
        self._ids = itertools.count()
        self.entries: list[ArchiveEntry] = []
        self.by_id: dict[int, ArchiveEntry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _is_duplicate(self, source_env: int, x: float, y: float) -> bool:
        recent = [e for e in self.entries[-self.cfg.dedup_check:]
                 if e.source_env == source_env]
        return any((e.x - x) ** 2 + (e.y - y) ** 2 < self.cfg.dedup_dist_m ** 2
                   for e in recent)

    def add(self, phys: Any, heavy_seen: list, heavy_seen_ticks: list,
           context_obs: np.ndarray, context_done: np.ndarray, direction: str,
           magnitude: float, source_env: int, source_step: int,
           lookback_offset: int, x: float, y: float) -> ArchiveEntry | None:
        """Insert unless it's a near-duplicate of a recent entry from the same
        source env. Returns the new entry (for broadcasting), or None."""
        if self._is_duplicate(source_env, x, y):
            return None
        entry = ArchiveEntry(next(self._ids), phys, heavy_seen, heavy_seen_ticks,
                             context_obs, context_done, direction, magnitude,
                             source_env, source_step, lookback_offset, x, y)
        self.entries.append(entry)
        self.by_id[entry.id] = entry
        if len(self.entries) > self.cfg.capacity:
            old = self.entries.pop(0)
            del self.by_id[old.id]
        return entry

    def get(self, entry_id: int) -> ArchiveEntry | None:
        return self.by_id.get(entry_id)

    def clear(self) -> None:
        """Drop every entry — a new exploiter practices its OWN hard states,
        not the previous one's. The id counter deliberately keeps running, so
        an id is never reused and a late reply carrying an old id can never be
        mistaken for a current entry."""
        self.entries.clear()
        self.by_id.clear()
