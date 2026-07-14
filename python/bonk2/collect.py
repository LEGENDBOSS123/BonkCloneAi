"""bonk2 shared-memory multiprocess env collection (EnvPool-style).

Worker processes each own a shard of LagEnvs and do everything per-tick
(physics, control, episode termination/reset, observation building). The main
process only sees decision-block granularity: one synchronization per
ACTION_REPEAT ticks, through preallocated shared arrays:

  obs   [E, 2, D]  observations for both seats at the block boundary
  act   [E, 2]     joint-action indices chosen by the main process
  brew  [E, 2]     per-seat reward accumulated over the block just simulated
  done  [E, 4]     [ended, dead0, dead1, timeout] if an episode ended in it

Cycle (two barriers):
  worker: write obs ─ B1 ─ (main: bookkeeping + inference + write act) ─ B2 ─
          read act, set decisions, simulate K ticks (accumulate brew/done,
          reset finished episodes; fresh envs idle out the block, keeping the
          global decision alignment) ─ repeat.

Workers ignore SIGINT; the main process owns shutdown via a shared stop flag.
All net inference stays in the main process — workers never see weights.
"""

import ctypes
import json
import multiprocessing as mp
import os
import random
import signal
import time

import numpy as np

from . import config as C


def _view(raw, dtype, shape):
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class _WorkerRecorder:
    """Replay writer for the first env of worker 0."""

    def __init__(self, out_dir, every):
        self.dir = out_dir
        self.every = every
        self.meta = {"tps": C.TPS, "actionRepeat": C.ACTION_REPEAT,
                     "inputLag": C.INPUT_LAG}
        self.frames = []
        self.seen = 0

    def frame(self, env, res):
        p0, p1 = env.sim.players
        a0, a1 = env.applied_actions()
        r3 = lambda v: round(v, 3)
        self.frames.append([r3(p0.pos[0]), r3(p0.pos[1]), 0,
                            r3(p1.pos[0]), r3(p1.pos[1]), 0, a0, a1])
        if not res["done"]:
            return
        self.seen += 1
        if self.seen % self.every == 0:
            result = ("draw" if res["timeout"] or all(res["dead"])
                      else "p1-wins" if res["dead"][1] else "p2-wins")
            path = os.path.join(self.dir, f"w0-game{self.seen}-{result}.json")
            with open(path, "w") as f:
                json.dump({"version": 2, **self.meta, "result": result,
                           "frames": self.frames}, f)
        self.frames = []


def worker_main(wid, offset, n_envs, map_path, raws, shapes, b_obs, b_act,
                stop_flag, seed, replay_cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    np.random.seed(seed)
    random.seed(seed)
    # Imports here so the spawned process pays them, not the parent fork point.
    from bonk.sim import BonkSim

    from .env import LagEnv

    obs = _view(raws["obs"], np.float32, shapes["obs"])
    act = _view(raws["act"], np.int64, shapes["act"])
    brew = _view(raws["brew"], np.float32, shapes["brew"])
    done = _view(raws["done"], np.float32, shapes["done"])

    with open(map_path) as f:
        map_json = json.load(f)
    envs = [LagEnv(BonkSim(map_json)) for _ in range(n_envs)]
    for env in envs:
        env.reset()

    recorder = _WorkerRecorder(**replay_cfg) if (wid == 0 and replay_cfg) else None

    sl = slice(offset, offset + n_envs)
    while True:
        for i, env in enumerate(envs):
            obs[offset + i, 0] = env.decision_state(0)
            obs[offset + i, 1] = env.decision_state(1)
        b_obs.wait()
        b_act.wait()
        if stop_flag.value:
            return

        for i, env in enumerate(envs):
            env.set_decision(0, int(act[offset + i, 0]))
            env.set_decision(1, int(act[offset + i, 1]))
        brew[sl] = 0.0
        done[sl] = 0.0
        for _ in range(C.ACTION_REPEAT):
            for i, env in enumerate(envs):
                if done[offset + i, 0]:
                    env.tick()  # fresh env idles out the rest of the block
                    continue
                res = env.tick()
                brew[offset + i, 0] += res["rewards"][0]
                brew[offset + i, 1] += res["rewards"][1]
                if recorder is not None and i == 0:
                    recorder.frame(env, res)
                if res["done"]:
                    done[offset + i] = [1.0, float(res["dead"][0]),
                                        float(res["dead"][1]), float(res["timeout"])]
                    env.reset()


class VecCollector:
    """Main-process handle: spawns workers, exposes the per-block sync API."""

    def __init__(self, workers, envs_per_worker, map_path,
                 replay_dir=None, replay_every=500):
        self.workers = workers
        self.E = workers * envs_per_worker
        self.k = C.ACTION_REPEAT
        ctx = mp.get_context("spawn")

        shapes = {
            "obs": (self.E, 2, C.STATE_DIM),
            "act": (self.E, 2),
            "brew": (self.E, 2),
            "done": (self.E, 4),
        }
        raws = {
            "obs": ctx.RawArray(ctypes.c_float, self.E * 2 * C.STATE_DIM),
            "act": ctx.RawArray(ctypes.c_int64, self.E * 2),
            "brew": ctx.RawArray(ctypes.c_float, self.E * 2),
            "done": ctx.RawArray(ctypes.c_float, self.E * 4),
        }
        self.obs = _view(raws["obs"], np.float32, shapes["obs"])
        self.act = _view(raws["act"], np.int64, shapes["act"])
        self.brew = _view(raws["brew"], np.float32, shapes["brew"])
        self.done = _view(raws["done"], np.float32, shapes["done"])

        self.b_obs = ctx.Barrier(workers + 1)
        self.b_act = ctx.Barrier(workers + 1)
        self.stop_flag = ctx.Value(ctypes.c_int, 0, lock=False)

        self.procs = []
        for w in range(workers):
            replay_cfg = None
            if w == 0 and replay_dir:
                replay_cfg = {"out_dir": str(replay_dir), "every": replay_every}
            p = ctx.Process(
                target=worker_main,
                args=(w, w * envs_per_worker, envs_per_worker, str(map_path),
                      raws, shapes, self.b_obs, self.b_act, self.stop_flag,
                      1000 + w, replay_cfg),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def sync_obs(self):
        """Wait for all workers' observations; brew/done come back as copies
        (obs stays a live view — copy rows you keep)."""
        self.b_obs.wait()
        return self.obs, self.brew.copy(), self.done.copy()

    def send_actions(self, actions: np.ndarray):
        self.act[:] = actions
        self.b_act.wait()  # releases workers to simulate the next block

    def stop(self):
        self.stop_flag.value = 1
        # Release anyone parked on the barriers so the flag gets seen.
        try:
            self.b_obs.wait(timeout=2)
            self.b_act.wait(timeout=2)
        except Exception:
            pass
        deadline = time.time() + 3
        for p in self.procs:
            p.join(timeout=max(0.1, deadline - time.time()))
        for p in self.procs:
            if p.is_alive():
                p.terminate()
