"""Shared-memory multiprocess env collection (EnvPool-style).

Worker processes each own a shard of `LagEnv`s and do everything per-tick:
physics, control, episode termination, reset, observation building. The main
process only ever sees decision-block granularity — one synchronization per
`action_repeat` ticks, through preallocated shared arrays::

    obs   [E, 2, state_dim]  observations for both seats at the block boundary
    act   [E, 2]             joint-action indices chosen by the main process
    done  [E, 6]             [ended, dead0, dead1, timeout, seat0_outcome,
                               archive_entry_id]

Cycle (two barriers)::

    worker: write obs - B1 - (main: bookkeeping + inference + write act) - B2 -
            read act, set decisions, simulate K ticks (record done, reset
            finished episodes) - repeat

All net inference stays in the main process; workers never see weights.
Workers ignore SIGINT — the main process owns shutdown via a shared stop flag.

Differences from `ppo7/collect.py`:

* **`brew` is gone.** ppo7 shipped a per-seat reward float alongside the done
  flags, then the trainer ignored the env's classes anyway (re-deriving them
  from the death/timeout flags) and re-scored draws afterward. Terminal values
  now live in exactly one place — the trainer — which is what stops reward
  values and critic atoms from silently drifting apart. `done[:, 4]` carries
  the outcome CLASS instead. This is sound only because rewards here are
  terminal-only; a future per-step reward would need the channel back.
* **the config is passed explicitly** to each worker rather than re-imported
  in the child. A spawned worker used to pick up whatever the config module
  said at import time in ITS process, so any parent-side mutation would
  silently not reach it.

**The value-swing archive** (`ppo9.archive`, `Ppo9Config.archive`) adds a
sixth `done` column plus three OPTIONAL IPC channels (inert unless
`archive_track_frac > 0`), all designed around one constraint: *never
continuously transfer a full physics snapshot* (~12KB, ~200us to capture —
measured — so doing this every cycle for every tracked env would be real,
avoidable throughput loss at the default 5120 envs / ~50k steps/s). Instead:

* **retention is local and free.** A worker retains physics for its OWN
  `archive_track_frac` share of envs in a plain in-process ring buffer — zero
  serialization, zero IPC, just Python object storage in the SAME process that
  already owns the physics.
* **the main process only ASKS when it actually needs one** — after it has
  independently confirmed (via `ppo9.archive.reconstruct`/`detect_swing`, over
  data it already collects for free from `act_step`) that a specific past
  decision of a specific tracked env is worth archiving, so this is a small,
  per-worker request queue rather than a per-cycle broadcast.

  Rarity is enforced by a per-env COOLDOWN in `Trainer._archive_detect`, not by
  `archive.value_threshold` alone: a qualifying swing stays inside the retained
  window for many cycles, so a threshold on its own re-requests the SAME swing
  every cycle. Measured at 64 tracked envs that was ~93 requests/cycle against
  a 64/cycle drain, which overran the bounded open-request map and starved the
  mechanism to a standstill — 99% of replies arrived to find their request
  already evicted, and insertions stopped entirely.
* **new entries propagate to workers incrementally** — one small message per
  actual insertion (also rare), never a periodic resend of the whole archive.
* **all three channels are `archive_active`-gated**, a new lock-free
  `ctx.Value` the main process sets from `league.phase == "exploiter"` each
  cycle (same pattern `spawn_frac` already uses) — a worker does zero archive
  bookkeeping outside an exploiter phase, and clears its local ring/local
  archive copy on every transition (data from a different critic/phase must
  never leak into either detection or a later restart).
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import queue
import random
import signal
import threading
import time
from typing import Any, NamedTuple

import numpy as np

from .config import EnvConfig

DONE_ENDED, DONE_DEAD0, DONE_DEAD1, DONE_TIMEOUT, DONE_OUTCOME, DONE_ARCHIVE = range(6)
DONE_WIDTH = 6
NO_ARCHIVE = -1.0     # DONE_ARCHIVE value meaning "this episode was a normal spawn"


def _view(raw, dtype, shape) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class _WorkerRecorder:
    """Replay writer for the first env of worker 0 only."""

    def __init__(self, out_dir: str, every: int, cfg: EnvConfig) -> None:
        self.dir = out_dir
        self.every = every
        self.meta = {"tps": cfg.engine.tps,
                     "actionRepeat": cfg.episode.action_repeat,
                     "inputLag": cfg.lag.self_lag}
        self.frames: list[list[Any]] = []
        self.seen = 0

    def frame(self, env, res) -> None:
        p0, p1 = env.sim.players
        a0, a1 = env.applied_actions()
        r3 = lambda v: round(v, 3)      # noqa: E731
        self.frames.append([r3(p0.pos[0]), r3(p0.pos[1]), 0,
                            r3(p1.pos[0]), r3(p1.pos[1]), 0, a0, a1])
        if not res.done:
            return
        self.seen += 1
        if self.seen % self.every == 0:
            result = ("draw" if res.timeout or all(res.dead)
                      else "p1-wins" if res.dead[1] else "p2-wins")
            path = os.path.join(self.dir, f"w0-game{self.seen}-{result}.json")
            with open(path, "w") as f:
                json.dump({"version": 2, **self.meta, "result": result,
                           "frames": self.frames}, f)
        self.frames = []


class ArchiveRequest(NamedTuple):
    """Main -> one worker: "send me env `global_env_idx`'s retained physics
    snapshot from `decisions_ago` decisions before its most recent tick."""
    global_env_idx: int
    decisions_ago: int
    request_id: int          # echoed back so the main process can match replies


class ArchiveSnapshot(NamedTuple):
    """Worker -> main, in reply to an `ArchiveRequest` (best-effort — omitted
    entirely if the requested offset has already aged out of the worker's own
    ring, which the main process treats as a silent miss, not an error)."""
    request_id: int
    global_env_idx: int
    phys: Any
    heavy_seen: list
    heavy_seen_ticks: list


class ArchiveBroadcast(NamedTuple):
    """Main -> all workers: one newly-inserted archive entry."""
    entry_id: int
    phys: Any
    heavy_seen: list
    heavy_seen_ticks: list


class _LocalPhysicsRing:
    """One tracked env's in-process physics history. Zero IPC cost to
    maintain — see the module docstring for why this is worker-local rather
    than pushed anywhere."""

    __slots__ = ("cap", "snaps", "t", "filled")

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.snaps: list[dict | None] = [None] * capacity
        self.t = 0
        self.filled = 0

    def push(self, snap: dict) -> None:
        self.snaps[self.t % self.cap] = snap
        self.t += 1
        self.filled = min(self.filled + 1, self.cap)

    def get(self, decisions_ago: int) -> dict | None:
        """The snapshot from `decisions_ago` decisions before the latest push,
        or None if that's either out of range or has already aged out."""
        if decisions_ago < 0 or decisions_ago >= self.filled:
            return None
        return self.snaps[(self.t - 1 - decisions_ago) % self.cap]

    def clear(self) -> None:
        self.t = 0
        self.filled = 0
        self.snaps = [None] * self.cap


def worker_main(wid: int, offset: int, n_envs: int, cfg: EnvConfig,
                raws: dict, shapes: dict, b_obs, b_act, stop_flag,
                seed: int, replay_cfg: dict | None, spawn_frac,
                archive_active, archive_track_frac: float, archive_history: int,
                archive_capacity: int,
                archive_spawn_prob: float, archive_request_q: "mp.Queue | None",
                archive_response_q: "mp.Queue | None",
                archive_broadcast_q: "mp.Queue | None") -> None:
    """One worker: owns `n_envs` envs at `offset` in the shared arrays.

    Archive args are all `None`/inert when `archive_track_frac<=0` (the
    default) — see `VecCollector.__init__`, which only constructs the queues
    at all when `cfg_archive.enabled`.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Seed BOTH streams: `random` drives lag draws and the per-decision draw
    # hazard, `np.random` drives spawn-pool sampling.
    np.random.seed(seed)
    random.seed(seed)
    # Imported here so the spawned process pays the cost, not the parent.
    from bonk3.simadapter import make_sim

    from .env import LagEnv
    from .layout import build_layout

    obs = _view(raws["obs"], np.float32, shapes["obs"])
    act = _view(raws["act"], np.int64, shapes["act"])
    done = _view(raws["done"], np.float32, shapes["done"])

    layout = build_layout(cfg.obs)      # rebuilt here: ndarrays never cross pickle

    def _sf() -> float:
        return float(spawn_frac.value)

    envs = [LagEnv(make_sim(cfg.engine.map_name, engine=cfg.engine.engine),
                   cfg, layout) for _ in range(n_envs)]
    for env in envs:
        env.reset(spawn_frac=_sf())

    recorder = (_WorkerRecorder(cfg=cfg, **replay_cfg)
                if (wid == 0 and replay_cfg) else None)

    # ── archive-local state, all inert (zero per-cycle cost) unless a global
    # index in this worker's own [offset, offset+n_envs) range is TRACKED.
    # Tracked = a FIXED, deterministic prefix of GLOBAL env indices, so no
    # coordination with the main process (or other workers) is needed to know
    # which envs are whose responsibility.
    tracked_local = ([] if archive_request_q is None else
                     [i for i in range(n_envs)
                      if (offset + i) < round(archive_track_frac * _total_envs(shapes))])
    rings: dict[int, _LocalPhysicsRing] = {i: _LocalPhysicsRing(archive_history)
                                           for i in tracked_local}
    local_archive: list[ArchiveBroadcast] = []
    was_active = False       # for edge-detecting an archive_active transition

    def _archive_now_active() -> bool:
        return bool(archive_active is not None and archive_active.value)

    def _drain_requests() -> None:
        if archive_request_q is None:
            return
        while True:
            try:
                req: ArchiveRequest = archive_request_q.get_nowait()
            except queue.Empty:
                return
            local_i = req.global_env_idx - offset
            if not (0 <= local_i < n_envs) or local_i not in rings:
                continue     # not one of ours (shouldn't happen; defensive)
            snap = rings[local_i].get(req.decisions_ago)
            if snap is not None and archive_response_q is not None:
                try:
                    archive_response_q.put_nowait(ArchiveSnapshot(
                        req.request_id, req.global_env_idx, snap["phys"],
                        snap["heavy_seen"], snap["heavy_seen_ticks"]))
                except queue.Full:
                    pass     # best-effort; a dropped reply just means no candidate

    def _drain_broadcasts() -> None:
        if archive_broadcast_q is None:
            return
        while True:
            try:
                msg: ArchiveBroadcast = archive_broadcast_q.get_nowait()
            except queue.Empty:
                return
            local_archive.append(msg)
            # SAME cap and SAME FIFO order as the main process's `Archive`, so
            # the two hold the same window. The old cap of 4096 was wrong on
            # both counts: main never broadcasts an eviction, so the worker
            # simply kept everything it was ever sent. At the live config that
            # is ~24 MB/worker (~193 MB across 8) of physics snapshots against
            # main's 3 MB, accumulating one broadcast at a time -- and it also
            # let a worker start an episode from an entry main had long since
            # evicted, whose id `_archive_seed_hidden` then cannot resolve, so
            # the recurrent state silently fell back to a zero that the
            # reset-anchoring design exists to avoid.
            if len(local_archive) > archive_capacity:
                local_archive.pop(0)

    sl = slice(offset, offset + n_envs)
    k = cfg.episode.action_repeat
    while True:
        for i, env in enumerate(envs):
            obs[offset + i, 0] = env.decision_state(0)
            obs[offset + i, 1] = env.decision_state(1)

        active = _archive_now_active()
        if active != was_active:
            # Phase transition: a ring/local archive from a different critic
            # (or a different exploiter entirely) must never seed a restart
            # or answer a request under the NEW phase.
            for r in rings.values():
                r.clear()
            local_archive.clear()
            was_active = active
        if active and rings:
            # ONE snapshot per DECISION-CYCLE (not per tick), taken here so it
            # reflects the SAME physics state `obs[]` above was just built
            # from — "decisions_ago" then means what it says. Cheap and
            # LOCAL: no serialization, just this process's own object storage.
            for i in rings:
                rings[i].push(envs[i].snapshot_physics())
            _drain_requests()
            _drain_broadcasts()

        try:
            b_obs.wait()
            b_act.wait()
        except threading.BrokenBarrierError:
            # `VecCollector.stop` aborts the barriers to release whoever is
            # parked on them. That is the normal shutdown path, not a fault —
            # exit quietly instead of dumping a traceback into the training log
            # on every single run, as ppo7 did.
            return
        if stop_flag.value:
            return

        for i, env in enumerate(envs):
            env.set_decision(0, int(act[offset + i, 0]))
            env.set_decision(1, int(act[offset + i, 1]))
        done[sl] = 0.0
        done[sl, DONE_ARCHIVE] = NO_ARCHIVE
        for _ in range(k):
            for i, env in enumerate(envs):
                if done[offset + i, DONE_ENDED]:
                    env.tick()      # fresh env idles out the rest of the block,
                    continue        # keeping the global decision alignment
                res = env.tick()
                if recorder is not None and i == 0:
                    recorder.frame(env, res)
                if res.done:
                    used_entry = NO_ARCHIVE
                    if (active and local_archive
                            and np.random.random() < archive_spawn_prob):
                        entry = local_archive[np.random.randint(len(local_archive))]
                        env.reset_from_archive({"phys": entry.phys,
                                                "heavy_seen": entry.heavy_seen,
                                                "heavy_seen_ticks": entry.heavy_seen_ticks})
                        used_entry = float(entry.entry_id)
                    else:
                        env.reset(spawn_frac=_sf())
                    done[offset + i] = [1.0, float(res.dead[0]), float(res.dead[1]),
                                        float(res.timeout), float(res.outcome[0]),
                                        used_entry]


def _total_envs(shapes: dict) -> int:
    return shapes["obs"][0]


class VecCollector:
    """Main-process handle: spawns workers, exposes the per-block sync API."""

    def __init__(self, cfg: EnvConfig, workers: int, envs_per_worker: int,
                 replay_dir: str | None = None, replay_every: int = 500,
                 archive_enabled: bool = False, archive_track_frac: float = 0.0,
                 archive_history: int = 128, archive_capacity: int = 512,
                 archive_spawn_prob: float = 0.0,
                 archive_queue_maxsize: int = 4096) -> None:
        from .layout import build_layout

        self.cfg = cfg
        self.workers = workers
        self.E = workers * envs_per_worker
        self.k = cfg.episode.action_repeat
        self.state_dim = build_layout(cfg.obs).state_dim
        ctx = mp.get_context("spawn")

        shapes = {"obs": (self.E, 2, self.state_dim),
                  "act": (self.E, 2),
                  "done": (self.E, DONE_WIDTH)}
        raws = {"obs": ctx.RawArray(ctypes.c_float, self.E * 2 * self.state_dim),
                "act": ctx.RawArray(ctypes.c_int64, self.E * 2),
                "done": ctx.RawArray(ctypes.c_float, self.E * DONE_WIDTH)}
        self.obs = _view(raws["obs"], np.float32, shapes["obs"])
        self.act = _view(raws["act"], np.int64, shapes["act"])
        self.done = _view(raws["done"], np.float32, shapes["done"])

        self.b_obs = ctx.Barrier(workers + 1)
        self.b_act = ctx.Barrier(workers + 1)
        self.stop_flag = ctx.Value(ctypes.c_int, 0, lock=False)
        # Read by every worker at every reset; the trainer moves it each cycle
        # to drive the spawn curriculum. Lock-free on purpose — a torn double
        # read at worst mis-spaces one episode.
        self.spawn_frac = ctx.Value(ctypes.c_double, 1.0, lock=False)

        # ── archive plumbing: entirely absent (None) unless enabled ────────
        self.archive_enabled = archive_enabled and archive_track_frac > 0.0
        # Exposed so a caller can assert the worker-side cap matches the main
        # archive's; nothing broadcasts an eviction, so this IS the bound.
        self.archive_capacity = archive_capacity
        self.archive_active = ctx.Value(ctypes.c_int, 0, lock=False) if self.archive_enabled else None
        self.archive_request_qs: list | None = None
        self.archive_response_q = None
        self.archive_broadcast_qs: list | None = None
        if self.archive_enabled:
            self.archive_request_qs = [ctx.Queue(maxsize=archive_queue_maxsize)
                                       for _ in range(workers)]
            self.archive_response_q = ctx.Queue(maxsize=archive_queue_maxsize)
            self.archive_broadcast_qs = [ctx.Queue(maxsize=archive_queue_maxsize)
                                         for _ in range(workers)]

        self.procs: list[mp.process.BaseProcess] = []
        for w in range(workers):
            replay_cfg = ({"out_dir": str(replay_dir), "every": replay_every}
                          if (w == 0 and replay_dir) else None)
            # RawArray/Barrier/Value/Queue must stay separate positional args:
            # the spawn start method special-cases them during pickling.
            p = ctx.Process(
                target=worker_main,
                args=(w, w * envs_per_worker, envs_per_worker, cfg,
                      raws, shapes, self.b_obs, self.b_act, self.stop_flag,
                      1000 + w, replay_cfg, self.spawn_frac,
                      self.archive_active, archive_track_frac, archive_history,
                      archive_capacity, archive_spawn_prob,
                      self.archive_request_qs[w] if self.archive_enabled else None,
                      self.archive_response_q if self.archive_enabled else None,
                      self.archive_broadcast_qs[w] if self.archive_enabled else None),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def sync_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Wait for every worker's observations.

        Returns:
            ``(obs[E,2,state_dim], done[E,6])``. **`obs` stays a LIVE VIEW** —
            the workers overwrite it during the next block, so copy any row you
            intend to keep. `done` is copied for you.
        """
        self.b_obs.wait()
        return self.obs, self.done.copy()

    def send_actions(self, actions: np.ndarray) -> None:
        """Publish ``[E,2]`` int64 actions and release the workers."""
        self.act[:] = actions
        self.b_act.wait()

    # ── archive: called by Trainer only when self.archive_enabled ──────────
    def set_archive_active(self, active: bool) -> None:
        if self.archive_active is not None:
            self.archive_active.value = 1 if active else 0

    def request_snapshot(self, global_env_idx: int, decisions_ago: int,
                         request_id: int, envs_per_worker: int) -> None:
        """Ask the OWNING worker for one env's retained physics — a rare,
        on-demand pull, never a per-cycle broadcast (see the module
        docstring)."""
        if self.archive_request_qs is None:
            return
        w = global_env_idx // envs_per_worker
        try:
            self.archive_request_qs[w].put_nowait(
                ArchiveRequest(global_env_idx, decisions_ago, request_id))
        except queue.Full:
            pass     # best-effort; the trainer just won't get a reply

    def drain_snapshots(self, max_items: int = 64) -> list[ArchiveSnapshot]:
        """Non-blocking drain of whatever archive-candidate replies have
        arrived since the last call."""
        if self.archive_response_q is None:
            return []
        out = []
        for _ in range(max_items):
            try:
                out.append(self.archive_response_q.get_nowait())
            except queue.Empty:
                break
        return out

    def broadcast_entry(self, entry_id: int, phys: Any, heavy_seen: list,
                        heavy_seen_ticks: list) -> None:
        """Push ONE newly-inserted archive entry to every worker — incremental,
        called only on an actual insertion, never on a fixed cadence."""
        if self.archive_broadcast_qs is None:
            return
        msg = ArchiveBroadcast(entry_id, phys, heavy_seen, heavy_seen_ticks)
        for q in self.archive_broadcast_qs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass     # a worker that falls behind just misses this one entry

    def stop(self) -> None:
        """Signal shutdown and reap the workers."""
        self.stop_flag.value = 1
        # Release anyone parked on a barrier so the flag actually gets seen.
        # `abort()` is immediate and deterministic; waiting with a timeout made
        # shutdown take seconds and still raced.
        try:
            self.b_obs.abort()
            self.b_act.abort()
        except Exception:
            pass
        deadline = time.time() + 3
        for p in self.procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        for p in self.procs:
            if p.is_alive():
                p.terminate()
