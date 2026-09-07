"""Horizontal-mirror data augmentation for observations and actions.

The map is horizontally symmetric, so every transition has a valid mirror twin:
negate x / vx (and the SIN half of their Fourier features), swap the left/right
key bits, and swap left<->right in the joint-action index. Training on both
doubles the effective data for free and forces the policy to be symmetric
rather than to learn each side separately.

The index tables themselves (`config.MIRROR_NEGATE`, `config.MIRROR_SWAPS`) are
built in config.py, because they depend on the observation layout, which is
conditional on `config.RECURRENT`. This module is only the transform.

Verified property: `mirror(fourier_expand(x)) == fourier_expand(mirror_raw(x))`
exactly, for both the recurrent (single-frame) and windowed (TCN) layouts.
"""

import numpy as np

from . import config as C


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Horizontal mirror of a single STATE_DIM observation. Heavy-meter features
    are symmetric and pass through unchanged."""
    out = obs.copy()
    out[C.MIRROR_NEGATE] *= -1
    for a, b in C.MIRROR_SWAPS:
        out[a], out[b] = out[b], out[a]
    return out


def mirror_obs_batch(states: np.ndarray) -> np.ndarray:
    """mirror_obs over an [N, STATE_DIM] array in three numpy ops (the per-row
    version dominates the update at scale)."""
    out = states.copy()
    out[:, C.MIRROR_NEGATE] *= -1
    for a, b in C.MIRROR_SWAPS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def mirror_action(index: int) -> int:
    """Swap left/right in the joint-action index; up/down/heavy unchanged."""
    lr, rest = index // 6, index % 6
    return (2 if lr == 1 else 1 if lr == 2 else 0) * 6 + rest


def mirror_action_batch(actions: np.ndarray) -> np.ndarray:
    """mirror_action over an [N] int array."""
    lr = actions // 6
    return np.where(lr == 1, 2, np.where(lr == 2, 1, 0)) * 6 + actions % 6
