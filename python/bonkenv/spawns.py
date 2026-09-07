"""Where episodes start: the survivable-spawn pool and the distance curriculum.

Both change the INITIAL STATE DISTRIBUTION only. Neither injects strategy, so
both stay inside the project's no-hand-written-rule-rewards constraint.
"""
from __future__ import annotations

import functools
import json

import numpy as np

from .config import CurriculumConfig

Position = tuple[float, float]


@functools.lru_cache(maxsize=8)
def _load_positions(path: str) -> np.ndarray:
    """``[n, 2]`` float32 survivable positions. Cached PER PATH.

    `ppo7/env.py` cached this in a module global keyed on nothing, so two
    configs with different maps in one process silently shared whichever pool
    loaded first. Keying the cache on the path removes that.
    """
    with open(path) as f:
        return np.asarray(json.load(f)["positions"], dtype=np.float32)


class SpawnPool:
    """Positions pre-probed by `gen_spawns` — every point is stable ground.

    Neither disc ever starts in a death trap, and "self" is spread across the
    whole map, which exercises both seats and the mirror augmentation.
    """

    def __init__(self, path: str, min_sep: float) -> None:
        self.positions = _load_positions(path)
        self.min_sep = min_sep

    def sample_pair(self) -> tuple[Position, Position]:
        """Two survivable positions at least `min_sep` apart.

        Uses the module-level `np.random` stream on purpose: workers seed it
        per process, so this inherits the worker's stream exactly as ppo7 did.
        Gives up after 16 rejections rather than loop on a degenerate pool.
        """
        pool = self.positions
        p0 = pool[np.random.randint(len(pool))]
        for _ in range(16):
            p1 = pool[np.random.randint(len(pool))]
            if np.hypot(*(p1 - p0)) >= self.min_sep:
                return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))
        return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))


class SpawnCurriculum:
    """Samples `spawn_frac`: 0.0 = discs start close, 1.0 = natural spawns.

    Sampled per episode rather than annealed to a single value, so every
    distance stays visited throughout and the far spawns arrive in the tail
    rather than being switched on abruptly. Two components:

    * a PERMANENT close-start floor (`close_prob`), which never anneals away —
      once the ramp fully opens, every game otherwise spawns far apart, and far
      apart the policy collapses to "stall when far" because chasing an evader
      is futile. The floor keeps winnable close fights coming so the kill skill
      is never lost.
    * the ramp itself, whose exponent relaxes from `1 + shape` toward 1
      (uniform) as `steps` are accumulated.
    """

    def __init__(self, cfg: CurriculumConfig) -> None:
        self.cfg = cfg

    def sample(self, env_steps: int, rng: np.random.Generator) -> float:
        """Draw one episode's spawn fraction in ``[0, 1]``."""
        cfg = self.cfg
        if not cfg.enabled:
            return 1.0
        if rng.random() < cfg.close_prob:
            return float(rng.random() ** cfg.close_shape)   # skewed close/mid
        ceil = min(1.0, env_steps / max(1, cfg.steps))
        beta = 1.0 + cfg.shape * (1.0 - ceil)
        return float(rng.random() ** beta)
