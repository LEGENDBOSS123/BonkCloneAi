"""RolloutBuffer: dense `[T, E]` on-policy storage for the learner seat.

Every env advances exactly ONE row per decision-cycle — a FiGAR-held row is
still a row — so the streams are dense `[T, E]` arrays and GAE vectorizes over
all E columns at once. (Per-env Python lists were the main-process bottleneck
at E ~= 2560.)

**The two-phase protocol.** A decision is made at cycle `t`, but whether it
ended the episode is only known after the workers have simulated the block,
which happens at cycle `t+1`. So each cycle runs, in order::

    commit(...)        # close the row opened last cycle, using THIS cycle's dones
    <inference>        # set_hidden(...) then set_learner(...) opens the next row

Two differences from `ppo7/rollout.py`:

* **No opponent seat.** ppo7 carried 16 parallel `b_o*`/`po_*` arrays for the
  mirror-match "both seats train" mode, which only the feedforward update ever
  read; the recurrent path always opened the seat as invalid. At production
  shape that was ~30 MB and a `[T,E,D]` copy per cycle, for nothing.
* **`b_r` is derived here, from the outcome class.** The env no longer ships a
  reward channel, so terminal values are defined in exactly one place and
  cannot drift away from the critic's atoms. The staller/exploiter revaluation
  is applied at commit time rather than patched into `b_r` afterward.
"""
from __future__ import annotations

import numpy as np


class RolloutBuffer:
    """`[T, E]` streams plus the single in-flight row."""

    def __init__(self, capacity: int, n_envs: int, state_dim: int,
                 gru_hidden: int) -> None:
        self.T, self.E, self.D = capacity, n_envs, state_dim
        self.H = gru_hidden
        self.t = 0                      # rows written so far this rollout

        T, E, D, H = capacity, n_envs, state_dim, gru_hidden
        self.b_s = np.zeros((T, E, D), np.float32)      # observation at the decision
        self.b_a = np.zeros((T, E), np.int64)           # joint action APPLIED
        self.b_lp = np.zeros((T, E), np.float32)        # sampled log-prob (free rows)
        self.b_v = np.zeros((T, E), np.float32)         # critic value at EVERY row
        self.b_r = np.zeros((T, E), np.float32)         # reward, derived at commit
        self.b_d = np.zeros((T, E), np.float32)         # done flag (0/1 as float)
        # Seat-0 outcome CLASS at a terminal row (0=win, 1=draw, 2=loss);
        # -1 elsewhere. The critic must label from the ACTUAL result, never
        # from nearest-atom-to-reward: draw and loss carry the same reward, so
        # that shortcut silently merged the two classes and the critic read
        # every lost position as a draw.
        self.b_oc = np.full((T, E), -1, np.int64)
        self.b_dur = np.zeros((T, E), np.int64)         # FiGAR duration INDEX
        # Free rows are real policy decisions; held rows carry a forced repeat
        # whose importance ratio would be off-policy garbage. The actor trains
        # on free rows only; the critic and GAE use every row.
        self.b_free = np.zeros((T, E), bool)
        self.b_staller = np.zeros((T, E), bool)         # opponent is a staller

        # Critic hidden BEFORE each row's step: the update re-forwards the
        # sequence from row 0, and the staller value baseline needs arbitrary
        # rows, so all T are kept.
        self.b_hc = np.zeros((T, E, H), np.float32)
        # The actor only ever needs the ROLLOUT-START hidden, so one row of it
        # is kept rather than T (ppo7 stored [T,E,H] and read index 0).
        self.ha0 = np.zeros((E, H), np.float32)

        # ── the in-flight row ──────────────────────────────────────────────
        self.p_valid = False            # no pending row until the first inference
        self.p_s = np.zeros((E, D), np.float32)
        self.p_a = np.zeros(E, np.int64)
        self.p_lp = np.zeros(E, np.float32)
        self.p_v = np.zeros(E, np.float32)
        self.p_dur = np.zeros(E, np.int64)
        self.p_free = np.zeros(E, bool)
        self.p_ha = np.zeros((E, H), np.float32)
        self.p_hc = np.zeros((E, H), np.float32)

    # ── writing ────────────────────────────────────────────────────────────
    def set_hidden(self, h_actor: np.ndarray, h_critic: np.ndarray) -> None:
        """Record the PRE-step recurrent hidden for the row about to open.

        Args:
            h_actor:  ``[E, H]`` actor hidden before this row's step.
            h_critic: ``[E, H]`` critic hidden before this row's step.
        """
        self.p_ha[:] = h_actor
        self.p_hc[:] = h_critic

    def set_learner(self, states: np.ndarray, actions: np.ndarray,
                    durations: np.ndarray, free: np.ndarray,
                    logps: np.ndarray, values: np.ndarray) -> None:
        """Open the next row.

        `logps` and `durations` are what the policy SAMPLED this cycle and are
        meaningful only where `free`. `values` is recorded at EVERY row,
        because GAE needs a value on every step.

        Args:
            states:    ``[E, D]``  observation fed to the net (incl. hold feature).
            actions:   ``[E]``     action actually applied (sampled or held).
            durations: ``[E]``     sampled duration index.
            free:      ``[E]``     this row was a real decision.
            logps:     ``[E]``     joint log-prob of the sampled (action, duration).
            values:    ``[E]``     critic value at this state.
        """
        self.p_s[:] = states
        self.p_a[:] = actions
        self.p_dur[:] = durations
        self.p_free[:] = free
        self.p_lp[:] = logps
        self.p_v[:] = values
        self.p_valid = True

    def commit(self, ended: np.ndarray, outcome: np.ndarray,
               staller_mask: np.ndarray, values: np.ndarray,
               staller_values: np.ndarray | None = None) -> int:
        """Write the open row into the streams and advance.

        The reward is derived HERE from the outcome class, so terminal values
        live in one place and cannot drift from the critic's atoms.

        Args:
            ended:          ``[E]`` bool, the episode ended on this row.
            outcome:        ``[E]`` int64 outcome class, -1 where still running.
            staller_mask:   ``[E]`` bool, this row's opponent is a staller.
            values:         ``[3]`` terminal value per class (win, draw, loss).
            staller_values: ``[3]`` values to use where `staller_mask`; when
                            omitted, `values` applies everywhere. This is how a
                            main scores its timeout against a graduated staller
                            as 0 (neutral) rather than -1, so its gradient
                            points at the kill instead of at avoiding the
                            staller.

        Returns:
            The index of the row just written.
        """
        t = self.t
        if t >= self.T:
            raise IndexError(f"rollout buffer full at capacity {self.T}")
        self.b_s[t] = self.p_s
        self.b_a[t] = self.p_a
        self.b_lp[t] = self.p_lp
        self.b_v[t] = self.p_v
        self.b_dur[t] = self.p_dur
        self.b_free[t] = self.p_free
        self.b_hc[t] = self.p_hc
        self.b_d[t] = ended
        self.b_oc[t] = outcome
        self.b_staller[t] = staller_mask
        if t == 0:
            # The sequence the update re-forwards starts here.
            self.ha0[:] = self.p_ha

        # outcome is -1 on non-terminal rows, which would index the LAST atom,
        # so gate on `ended` rather than relying on the index.
        idx = np.where(ended, outcome, 0)
        base = np.asarray(values, np.float32)[idx]
        if staller_values is not None:
            alt = np.asarray(staller_values, np.float32)[idx]
            base = np.where(staller_mask, alt, base)
        self.b_r[t] = np.where(ended, base, 0.0)

        self.t = t + 1
        self.p_valid = False
        return t

    # ── lifecycle ──────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Start a new rollout, KEEPING the in-flight row.

        Called after an update: the open row belongs to the same learner and
        becomes row 0 of the next rollout. Rows beyond `t` are stale — always
        slice `[:t]`.
        """
        self.t = 0

    def discard(self) -> None:
        """Throw away everything, in-flight row included.

        Called on a phase switch: the open row belongs to the OLD learner.
        """
        self.t = 0
        self.p_valid = False

    def learner_steps(self) -> int:
        """Rows written so far, counted across environments."""
        return self.t * self.E
