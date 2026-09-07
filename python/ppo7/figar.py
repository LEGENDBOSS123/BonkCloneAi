"""FiGAR hold state: one seat's "which action is latched, for how much longer".

The actor emits BOTH a joint action and a HOLD DURATION (config.DURATIONS, in
decision-cycles). The sampled action is then forced to repeat for that many
cycles before the seat decides again. This gives coherent, temporally-extended
exploration -- one sample means "hold right for 16 cycles" -- that per-cycle
noise never produces, and lets the agent pick its own timescale.

The net still runs EVERY cycle (its recurrent memory must advance in lockstep
with the rollout buffer); the hold only gates which sampled action is applied
and whether the row counts as a FREE decision the actor may train on.

One tracker per seat, vectorized across all E envs. Every method takes an
optional `idx` so a seat can be driven for a subset of envs -- the opponent seat
is driven per opponent-net group, and each group is a different subset.
"""

import numpy as np

from . import config as C


class HoldTracker:
    def __init__(self, n_envs: int):
        self.hold = np.zeros(n_envs, dtype=np.int64)   # cycles left on the latch
        self.held = np.zeros(n_envs, dtype=np.int64)   # the action being held
        self._durs = np.asarray(C.DURATIONS, dtype=np.int64)
        self.maxdur = float(max(C.DURATIONS))

    def feature(self, idx=None) -> np.ndarray:
        """[n, 1] remaining hold, normalised to [0, 1]. Appended to the obs as
        the net's last input feature: it makes a mid-hold state Markov for the
        critic ("locked into action 3 for 63 more" reads differently from
        "free"). Must be read BEFORE `apply` for the current cycle."""
        h = self.hold if idx is None else self.hold[idx]
        return (h.astype(np.float32) / self.maxdur)[:, None]

    def free_mask(self, idx=None) -> np.ndarray:
        """[n] bool — rows whose hold has expired, i.e. real policy decisions.
        Only these may train the actor: a held row carries a forced repeat whose
        importance ratio would be off-policy garbage."""
        return (self.hold if idx is None else self.hold[idx]) == 0

    def latch(self, idx, actions, durations) -> None:
        """Start a new hold on `idx` (which must already be free): remember the
        action and how many cycles to repeat it for."""
        self.held[idx] = actions
        self.hold[idx] = self._durs[durations]

    def advance(self, idx=None) -> np.ndarray:
        """The actions to execute this cycle; advances the clock by one."""
        sel = slice(None) if idx is None else idx
        applied = self.held[sel].copy()
        self.hold[sel] -= 1
        return applied

    def apply(self, actions, durations, idx=None):
        """free_mask + latch + advance, for the common case where the net was
        run on EVERY row of the subset (held rows sample an action that is then
        discarded). Returns (applied [n], free [n] bool)."""
        free = self.free_mask(idx)
        rows = np.nonzero(free)[0] if idx is None else idx[free]
        self.latch(rows, actions[free], durations[free])
        return self.advance(idx), free

    def reset(self, mask) -> None:
        """A new episode starts FREE (no hold carried across the boundary)."""
        self.hold[mask] = 0
