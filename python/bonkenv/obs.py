"""Raw-frame index constants and the Fourier expansion.

Kept separate from `env.py` so the observation FORMAT can be read (and tested)
without constructing a physics sim.
"""
from __future__ import annotations

import numpy as np

from .layout import ObsLayout

# Offsets into the raw frame. Every mirror table and every downstream feature
# index is stated relative to these, so a layout change has one place to edit.
SELF_OFF = 0        # 12 dims: x, y, vx, vy, heavy(masked), up/down/left/right,
OPP_OFF = 12        # 12 dims: same, but as PREDICTED under rollback netcode
REL_OFF = 24        # 4 dims:  dx, dy, dvx, dvy  (opponent minus self)
PEND_OFF = 28       # 5 dims:  own last DECIDED action bits
BLOCK_DIM = 12      # width of a per-player block


def fourier_expand(raw: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """NeRF-style positional features for one raw frame.

    Args:
        raw:    ``[raw_frame_dim]`` float32 raw observation.
        layout: geometry (which dims get expanded, at which frequencies).

    Returns:
        ``[fourier_block]`` float32, laid out per base dim as
        ``sin_0, cos_0, sin_1, cos_1, ...``. The interleaving is required by
        the mirror tables — see `layout.build_layout`.
    """
    ang = np.outer(raw[layout.fourier_base], layout.fourier_freqs)
    blk = np.empty((len(layout.fourier_base), layout.fourier_feats), dtype=np.float32)
    blk[:, 0::2] = np.sin(ang)
    blk[:, 1::2] = np.cos(ang)
    return blk.reshape(-1)
