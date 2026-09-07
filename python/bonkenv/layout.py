"""Observation geometry, derived from `ObsConfig` as VALUES, not import-time globals.

`ppo7/config.py` computed `STATE_DIM` and the mirror index tables at import
time from a module-level `RECURRENT` flag, which is why its tests had to
`importlib.reload` half the package to change layout. Here the same quantities
come out of a pure, cached function, so two different layouts can coexist in
one process.

The mirror tables are the subtle part. A horizontal flip negates x and vx. For
a Fourier-expanded dim ``v -> -v``::

    sin(pi 2^k (-v)) = -sin(pi 2^k v)      NEGATE
    cos(pi 2^k (-v)) =  cos(pi 2^k v)      keep

so the flip must also negate the SIN half of each negated dim's Fourier block
and leave the COS half alone. That is exact only because the block interleaves
``sin_0, cos_0, sin_1, cos_1, ...`` per base dim — a refactor to
``all_sins || all_coses`` would break it SILENTLY. `tests/test_layout.py`
gates the property both ways.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .config import ObsConfig


@dataclass(frozen=True, eq=False)   # eq=False: ndarray fields break a generated __eq__
class ObsLayout:
    """Derived observation geometry. All index arrays are `np.intp`.

    Attributes:
        raw_frame_dim:  width of the raw frame, before expansion.
        fourier_feats:  entries per base dim = 2 * fourier_l (sin+cos per octave).
        fourier_block:  total Fourier width = len(base_dims) * fourier_feats.
        fourier_offset: where the Fourier block starts in the observation.
        state_dim:      full observation width = raw_frame_dim + fourier_block.
        fourier_base:   [n_base] indices into the raw frame that get expanded.
        fourier_freqs:  [fourier_l] angular frequencies pi * 2^k.
        mirror_negate:  [n_neg] observation indices whose sign flips on mirror.
        mirror_swap_a:  [n_swap] left indices of the left<->right key swaps.
        mirror_swap_b:  [n_swap] matching right indices.
    """

    raw_frame_dim: int
    fourier_feats: int
    fourier_block: int
    fourier_offset: int
    state_dim: int
    fourier_base: np.ndarray
    fourier_freqs: np.ndarray
    mirror_negate: np.ndarray
    mirror_swap_a: np.ndarray
    mirror_swap_b: np.ndarray


@functools.lru_cache(maxsize=8)
def build_layout(obs: ObsConfig) -> ObsLayout:
    """Derive the observation geometry for `obs`.

    Cached on the (hashable, frozen) `ObsConfig`, so workers rebuild it for
    free instead of receiving ndarray-bearing objects through pickle.
    """
    feats = 2 * obs.fourier_l
    block = len(obs.fourier_base_dims) * feats
    offset = obs.raw_frame_dim
    state_dim = obs.raw_frame_dim + block

    freqs = (np.pi * (2.0 ** (obs.fourier_k_start
                              + np.arange(obs.fourier_l)))).astype(np.float32)
    base = np.asarray(obs.fourier_base_dims, dtype=np.intp)

    # Raw negations, plus the SIN entries of every negated base dim's block.
    neg_set = set(obs.mirror_negate_raw)
    fourier_sin = [offset + j * feats + 2 * k
                   for j, d in enumerate(obs.fourier_base_dims) if d in neg_set
                   for k in range(obs.fourier_l)]
    negate = np.asarray(list(obs.mirror_negate_raw) + fourier_sin, dtype=np.intp)

    swaps = np.asarray(obs.mirror_swaps_raw, dtype=np.intp).reshape(-1, 2)
    return ObsLayout(
        raw_frame_dim=obs.raw_frame_dim,
        fourier_feats=feats,
        fourier_block=block,
        fourier_offset=offset,
        state_dim=state_dim,
        fourier_base=base,
        fourier_freqs=freqs,
        mirror_negate=negate,
        mirror_swap_a=swaps[:, 0].copy(),
        mirror_swap_b=swaps[:, 1].copy(),
    )
