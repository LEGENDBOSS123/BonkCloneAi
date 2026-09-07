"""A `VecCollector`-compatible collector with envs living directly in this
process, not in worker subprocesses.

`Trainer` only ever talks to a collector through `sync_obs()`/`send_actions()`/
`.E`/`.k`/`.spawn_frac` -- it never touches an env directly, by design (`
collect.py`'s own docstring: "All net inference stays in the main process --
workers never see weights"). That design is exactly right for real training,
and exactly wrong for an experiment that needs to run `LookaheadPolicy`
against the ACTUAL live env for a handful of decisions: the real
`VecCollector`'s envs live in separate OS processes and are never reachable
from the trainer's process at all.

This class is the deliberate exception: same public interface as
`VecCollector`, so an unmodified `Trainer` runs correctly against it, but
`.envs` is a plain Python list the SAME process can reach into. Single-
process only -- there is no multiprocessing here, so it does not scale to
real training sizes (`VecCollector` remains what actual training runs use);
it exists for exactly this kind of single-process, direct-env-access
prototype (see `experiment_search_finetune.py`).
"""

import types

import numpy as np

from bonk3.simadapter import make_sim

from . import config as C
from .env import LagEnv


class SingleProcessCollector:
    def __init__(self, n_envs: int):
        self.E = n_envs
        self.k = C.ACTION_REPEAT
        self.spawn_frac = types.SimpleNamespace(value=1.0)
        self.envs = [LagEnv(make_sim(C.MAP_NAME, engine=C.ENGINE))
                    for _ in range(n_envs)]
        for e in self.envs:
            e.reset(spawn_frac=self.spawn_frac.value)
        self._obs = np.zeros((n_envs, 2, C.STATE_DIM), dtype=np.float32)
        self._brew = np.zeros((n_envs, 2), dtype=np.float32)
        self._done = np.zeros((n_envs, 4), dtype=np.float32)
        self._refresh_obs()

    def _refresh_obs(self) -> None:
        for i, e in enumerate(self.envs):
            self._obs[i, 0] = e.decision_state(0)
            self._obs[i, 1] = e.decision_state(1)

    def sync_obs(self):
        """Matches `VecCollector.sync_obs`: obs is already current (refreshed
        at the end of the previous `send_actions`, or at construction);
        brew/done belong to the block that just finished."""
        return self._obs, self._brew.copy(), self._done.copy()

    def send_actions(self, actions: np.ndarray) -> None:
        """Matches `worker_main`'s per-block loop exactly, including an env
        that finishes mid-block idling out the rest of it (ticked, but its
        reward/outcome no longer accumulated) rather than being reset early."""
        for i, e in enumerate(self.envs):
            e.set_decision(0, int(actions[i, 0]))
            e.set_decision(1, int(actions[i, 1]))
        self._brew[:] = 0.0
        self._done[:] = 0.0
        sf = float(self.spawn_frac.value)
        for _ in range(self.k):
            for i, e in enumerate(self.envs):
                if self._done[i, 0]:
                    e.tick()             # fresh env idles out the rest of the block
                    continue
                res = e.tick()
                self._brew[i, 0] += res["rewards"][0]
                self._brew[i, 1] += res["rewards"][1]
                if res["done"]:
                    self._done[i] = [1.0, float(res["dead"][0]),
                                     float(res["dead"][1]), float(res["timeout"])]
                    e.reset(spawn_frac=sf)
        self._refresh_obs()

    def stop(self) -> None:
        pass
