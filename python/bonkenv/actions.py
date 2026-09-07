"""The joint action space and the outcome class enum.

An action is one index into the 3x3x2 product of (left/right) x (up/down) x
(heavy), packed as ``lr * 6 + rest`` — an encoding `mirror_action` depends on.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from bonk.sim import index_to_keys       # noqa: F401  (re-exported)

NUM_ACTIONS = 18        # 3 lr x 3 ud x 2 heavy
IDLE = 0


class Outcome(IntEnum):
    """Seat-0's terminal result. The ORDER is load-bearing.

    It indexes the categorical critic's atom vector directly, so ``WIN=0,
    DRAW=1, LOSS=2`` must match the learner's atom ordering everywhere.

    This deliberately differs from `ppo7/env.py`'s module constants, which were
    ``OUT_WIN, OUT_LOSS, OUT_DRAW = 0, 1, 2`` — i.e. draw and loss transposed
    relative to `ppo7/rollout.py`, `targets.py` and `trainer.py`, which all use
    win/draw/loss. ppo7 got away with the contradiction only because the env's
    outcome value was discarded in the worker and the trainer re-derived the
    class from the death/timeout flags. Plumbing the env's value through would
    have silently swapped two of the critic's three classes, with no error and
    (since draw and loss share a reward today) not even a reward change. One
    enum, one ordering, used on both sides of the boundary.
    """

    WIN = 0
    DRAW = 1
    LOSS = 2
    NONE = -1           # non-terminal; never a training label


def mirror_action(index: int) -> int:
    """Swap left/right in a joint-action index; up/down/heavy unchanged."""
    lr, rest = index // 6, index % 6
    return (2 if lr == 1 else 1 if lr == 2 else 0) * 6 + rest


def mirror_action_batch(actions: np.ndarray) -> np.ndarray:
    """`mirror_action` over an ``[N]`` int array -> ``[N]``."""
    lr = actions // 6
    return np.where(lr == 1, 2, np.where(lr == 2, 1, 0)) * 6 + actions % 6
