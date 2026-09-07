"""The on-policy rollout buffer: flat [T, E] streams plus the in-flight row.

Every env advances exactly one row per decision-cycle (holds are completed in
lockstep -- a held row is still a row), so the streams are dense `[T, E]` arrays
and GAE vectorizes across all E columns. Per-env Python lists were the
main-process bottleneck at E ~ 2560.

TWO-PHASE COMMIT. A decision is made at cycle t but its reward only exists after
the workers have simulated the block, which happens at cycle t+1. So each cycle:

    add_reward(brew)   fold the block's reward into the in-flight row
    commit(...)        write the completed row to [t] and advance t
    <inference>        set_learner(...) / set_opponent(...) open the next row

`p_*` is the learner seat's in-flight row, `po_*` the opponent seat's. The
opponent stream is only valid where `po_m` is set (i.e. the opponent was the
LIVE current model, a mirror match, so its rows are on-policy for the learner).

Recurrent mode also stores `p_ha`/`p_hc`: each row's minGRU hidden state as it
was BEFORE that row's step, so the update can re-forward the sequence from the
stored start hidden after the parameters have drifted.
"""

import numpy as np

from . import config as C


class RolloutBuffer:
    def __init__(self, capacity: int, n_envs: int, state_dim: int,
                 recurrent: bool = False, gru_hidden: int = 0):
        T, E, D = capacity, n_envs, state_dim
        self.T, self.E, self.D = T, E, D
        self.recurrent = recurrent
        self.t = 0                      # rows written so far this rollout

        # ── learner seat ────────────────────────────────────────────────────
        self.b_s = np.zeros((T, E, D), dtype=np.float32)
        self.b_a = np.zeros((T, E), dtype=np.int64)
        self.b_lp = np.zeros((T, E), dtype=np.float32)
        self.b_v = np.zeros((T, E), dtype=np.float32)
        self.b_r = np.zeros((T, E), dtype=np.float32)
        self.b_d = np.zeros((T, E), dtype=np.float32)
        # Seat-0 outcome CLASS per terminal row (0=win, 1=draw, 2=loss; -1 =
        # non-terminal). The categorical critic labels from THIS actual result,
        # not from nearest-atom-to-reward -- which silently merges draw and loss
        # whenever their rewards are equal (they are, both -1).
        self.b_oc = np.full((T, E), -1, dtype=np.int64)
        # FiGAR: the hold-duration index chosen at each row, and whether the row
        # was a FREE decision. The actor trains only on FREE rows; the critic and
        # GAE use every row.
        self.b_dur = np.zeros((T, E), dtype=np.int64)
        self.b_free = np.zeros((T, E), dtype=bool)
        # Per-row flag: the learner's opponent on this row is a STALLER
        # exploiter. Used to score the main's timeout as 0 (neutral) instead of
        # -1, and to re-weight the value baseline to match.
        self.b_staller = np.zeros((T, E), dtype=bool)

        # ── opponent seat (valid only where b_om) ───────────────────────────
        self.b_os = np.zeros((T, E, D), dtype=np.float32)
        self.b_oa = np.zeros((T, E), dtype=np.int64)
        self.b_olp = np.zeros((T, E), dtype=np.float32)
        self.b_ov = np.zeros((T, E), dtype=np.float32)
        self.b_or = np.zeros((T, E), dtype=np.float32)
        self.b_om = np.zeros((T, E), dtype=bool)
        self.b_odur = np.zeros((T, E), dtype=np.int64)
        self.b_ofree = np.zeros((T, E), dtype=bool)

        # ── in-flight row ───────────────────────────────────────────────────
        self.p_valid = False            # no pendings until the first inference
        self.p_s = np.zeros((E, D), dtype=np.float32)
        self.p_a = np.zeros(E, dtype=np.int64)
        self.p_lp = np.zeros(E, dtype=np.float32)
        self.p_v = np.zeros(E, dtype=np.float32)
        self.p_r = np.zeros(E, dtype=np.float32)
        self.p_dur = np.zeros(E, dtype=np.int64)
        self.p_free = np.zeros(E, dtype=bool)
        self.po_s = np.zeros((E, D), dtype=np.float32)
        self.po_a = np.zeros(E, dtype=np.int64)
        self.po_lp = np.zeros(E, dtype=np.float32)
        self.po_v = np.zeros(E, dtype=np.float32)
        self.po_r = np.zeros(E, dtype=np.float32)
        self.po_m = np.zeros(E, dtype=bool)
        self.po_dur = np.zeros(E, dtype=np.int64)
        self.po_free = np.zeros(E, dtype=bool)

        # ── recurrent hidden state ──────────────────────────────────────────
        if recurrent:
            H = gru_hidden
            self.p_ha = np.zeros((E, H), dtype=np.float32)
            self.p_hc = np.zeros((E, H), dtype=np.float32)
            self.b_ha = np.zeros((T, E, H), dtype=np.float32)
            self.b_hc = np.zeros((T, E, H), dtype=np.float32)

    # ── writing ─────────────────────────────────────────────────────────────
    def add_reward(self, brew) -> None:
        """Fold the just-simulated block's per-seat reward into the open row."""
        self.p_r += brew[:, 0]
        self.po_r += brew[:, 1]

    def commit(self, ended, outcome, staller_mask) -> int:
        """Write the in-flight row into the streams. Returns its index, so the
        caller can still patch per-row rewards (the staller / exploiter draw
        overrides) before the row is used."""
        t = self.t
        self.b_s[t] = self.p_s
        self.b_a[t] = self.p_a
        self.b_lp[t] = self.p_lp
        self.b_v[t] = self.p_v
        self.b_r[t] = self.p_r
        self.b_d[t] = ended
        self.b_oc[t] = outcome
        self.b_dur[t] = self.p_dur
        self.b_free[t] = self.p_free
        self.b_staller[t] = staller_mask
        self.b_os[t] = self.po_s
        self.b_oa[t] = self.po_a
        self.b_olp[t] = self.po_lp
        self.b_ov[t] = self.po_v
        self.b_or[t] = self.po_r
        self.b_om[t] = self.po_m
        self.b_odur[t] = self.po_dur
        self.b_ofree[t] = self.po_free
        if self.recurrent:
            self.b_ha[t] = self.p_ha      # hidden BEFORE this row's step
            self.b_hc[t] = self.p_hc
        self.t = t + 1
        self.p_valid = False
        return t

    def set_learner(self, states, actions, durations, free, logps, values) -> None:
        """Open a new in-flight row for the learner seat.

        `logps` and `durations` are what the policy SAMPLED this cycle and are
        only meaningful where `free`; `values` is recorded at EVERY row because
        GAE needs a value on every step of the stream.
        """
        self.p_s[:] = states
        self.p_a[:] = actions
        self.p_dur[:] = durations
        self.p_free[:] = free
        self.p_lp[:] = logps
        self.p_v[:] = values
        self.p_r[:] = 0.0
        self.p_valid = True

    def open_opponent(self, valid=False) -> None:
        """Open the opponent seat's in-flight row. `valid` [E] bool marks the
        envs whose opponent is the LIVE current model (a mirror match), the only
        ones whose rows are on-policy for the learner. Pass False to mark the
        whole seat unused -- no mirror match, or the recurrent path, which trains
        the learner seat only."""
        self.po_m[:] = valid
        self.po_r[:] = 0.0
        self.po_free[:] = False

    def set_opponent(self, idx, states, actions, durations, free, logps,
                     values) -> None:
        """Fill the opponent seat's in-flight row for the `idx` subset of envs
        (only the mirror-match envs are collected)."""
        self.po_s[idx] = states
        self.po_a[idx] = actions
        self.po_dur[idx] = durations
        self.po_free[idx] = free
        self.po_lp[idx] = logps
        self.po_v[idx] = values

    def set_hidden(self, h_actor, h_critic) -> None:
        """Record the pre-step minGRU hidden for the open row."""
        self.p_ha[:] = h_actor
        self.p_hc[:] = h_critic

    # ── lifecycle ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Drop all in-flight rows: the rollout restarts from row 0. Used both
        after an update (the pendings stay live and become row 0) and on a phase
        switch (where the caller also invalidates the pendings)."""
        self.t = 0

    def discard(self) -> None:
        """Phase changed: the open row belongs to the OLD learner, so throw it
        away along with the rows already written."""
        self.t = 0
        self.p_valid = False
        self.po_m[:] = False

    def learner_steps(self) -> int:
        return self.t * self.E
