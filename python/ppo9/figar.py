"""FiGAR hold tracking: which action is latched for a seat, and for how long.

The actor emits a joint action AND a hold duration (an index into
`FigarConfig.durations`, in decision-cycles); the sampled action is then forced
to repeat for that many cycles before the seat decides again.

Two things the net still does every cycle regardless of the hold: it runs, and
its recurrent memory advances. The hold gates only *which sampled action is
applied* and *whether the row counts as a free decision*. Held rows carry a
forced repeat whose importance ratio would be off-policy garbage, so the actor
trains on free rows only — while the critic and GAE use every row.
"""
from __future__ import annotations

import numpy as np


class HoldTracker:
    """One seat's latch state, vectorized over `n_envs` environments.

    Every method takes an optional `idx` (an integer array of env indices) so a
    seat can be driven for a subset — the opponent seat is stepped once per
    distinct opponent net, each a different subset of envs.
    """

    def __init__(self, n_envs: int, durations: tuple[int, ...]) -> None:
        self.n_envs = n_envs
        self.hold = np.zeros(n_envs, dtype=np.int64)    # cycles left on the latch
        self.held = np.zeros(n_envs, dtype=np.int64)    # the action being held
        self._durs = np.asarray(durations, dtype=np.int64)
        self.maxdur = float(max(durations))

    def feature(self, idx: np.ndarray | None = None) -> np.ndarray:
        """``[n, 1]`` float32 remaining hold, normalized to ``[0, 1]``.

        Appended to the observation as the net's LAST input feature (hence
        `agent_state_dim = state_dim + 1`). It is what makes a mid-hold state
        Markov for the critic: "locked into action 3 for 63 more cycles" must
        not read the same as "free to decide".

        Must be read BEFORE `apply` for the current cycle.
        """
        sel = slice(None) if idx is None else idx
        return (self.hold[sel] / self.maxdur).astype(np.float32)[:, None]

    def free_mask(self, idx: np.ndarray | None = None) -> np.ndarray:
        """``[n]`` bool — rows whose hold expired, i.e. real policy decisions."""
        sel = slice(None) if idx is None else idx
        return self.hold[sel] == 0

    def latch(self, idx: np.ndarray, actions: np.ndarray,
              durations: np.ndarray) -> None:
        """Start a new hold on `idx`. `durations` are INDICES into `durations`.

        Precondition (not checked): every row in `idx` is currently free.
        """
        self.held[idx] = actions
        self.hold[idx] = self._durs[durations]

    def advance(self, idx: np.ndarray | None = None) -> np.ndarray:
        """``[n]`` int64 actions to execute now; decrements every counter.

        Because the decrement is unconditional, a latch of `d` cycles yields
        exactly `d` applied cycles.
        """
        sel = slice(None) if idx is None else idx
        out = self.held[sel].copy()
        self.hold[sel] -= 1
        return out

    def apply(self, actions: np.ndarray, durations: np.ndarray,
              idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Latch newly-free rows, then advance. Returns ``(applied, free)``.

        The composite for the common case where the net ran on EVERY row of the
        subset: held rows still sample an action, which is then discarded.
        """
        free = self.free_mask(idx)
        rows = np.nonzero(free)[0] if idx is None else idx[free]
        self.latch(rows, actions[free], durations[free])
        return self.advance(idx), free

    def reset(self, mask: np.ndarray) -> None:
        """Clear holds where `mask` — a new episode always starts free."""
        self.hold[mask] = 0
