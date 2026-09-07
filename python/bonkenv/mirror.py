"""Horizontal-mirror augmentation for observations.

The map is horizontally symmetric, so every transition has a valid mirror twin:
negate x/vx (and the sin half of their Fourier features) and swap the
left/right key bits. Training on both halves doubles the effective data for
free and forces a symmetric policy instead of one that learns each side
separately.

Unlike `ppo7/mirror.py`, the index tables are passed in (see
`layout.ObsLayout`) rather than read from a module-level config, so a mirror
is well defined for whichever layout the caller is actually using.
"""
from __future__ import annotations

import numpy as np

from .layout import ObsLayout


def mirror_obs(obs: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """Mirror a single ``[state_dim]`` observation -> ``[state_dim]``.

    Heavy-meter and up/down features are symmetric and pass through unchanged.
    """
    out = obs.copy()
    out[layout.mirror_negate] *= -1
    a, b = layout.mirror_swap_a, layout.mirror_swap_b
    out[a], out[b] = out[b].copy(), out[a].copy()
    return out


def mirror_obs_batch(states: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """`mirror_obs` over ``[N, state_dim]`` -> ``[N, state_dim]``, vectorized.

    The per-row version dominates the update at scale, so this stays three
    numpy ops regardless of how many swap pairs there are.
    """
    out = states.copy()
    out[:, layout.mirror_negate] *= -1
    a, b = layout.mirror_swap_a, layout.mirror_swap_b
    out[:, a], out[:, b] = out[:, b].copy(), out[:, a].copy()
    return out
