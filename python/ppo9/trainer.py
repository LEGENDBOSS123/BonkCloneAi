"""Trainer: one decision-cycle of collection, and the update that follows.

A cycle is ingest -> infer -> (sometimes) update:

* **ingest** — close out the row opened last cycle using the dones the workers
  just produced, derive its reward from the outcome class, and run per-episode
  league bookkeeping.
* **infer** — step the learner and every opponent group, apply FiGAR holds,
  open the next row.
* **update** — once `rollout_steps` rows have accumulated, build the targets
  and hand them to the agent.

Only the LEARNER seat is collected. Seat 1 still acts as a full opponent with
its own carried recurrent memory, but its rows are never trained on: ppo7's
mirror-match "both seats train" mode existed only in the feedforward path, and
carrying its 16 parallel buffers cost ~30 MB and a `[T,E,D]` copy per cycle for
nothing. That is a deliberate one-way door.
"""
from __future__ import annotations

import itertools
import time
from typing import Any, Callable

import numpy as np
import torch

from bonkenv import DONE_ARCHIVE, DONE_ENDED, DONE_OUTCOME, NO_ARCHIVE, Outcome
from bonkenv import SELF_OFF, SpawnCurriculum
from bonkenv import mirror_action_batch, mirror_obs_batch

from . import targets
from .agent import PPOAgent
from .archive import (Archive, ValueTracker, detect_swing, last_reset_index,
                      replay_from_reset)
from .config import Ppo9Config
from .figar import HoldTracker
from .league import League
from .mingru import MinGRUNet
from .rollout import RolloutBuffer
from .schedules import duration_entropy_coef, entropy_coef_at, exploiter_entropy_coef

LEARNER, OPPONENT = 0, 1


class Trainer:
    """Owns collection, the rollout buffer, the league, and the update."""

    def __init__(self, collector: Any, agent: PPOAgent, cfg: Ppo9Config,
                 agent_factory: Callable[..., PPOAgent] | None = None) -> None:
        self.coll = collector
        self.agent = agent                  # the MAIN agent — always what gets saved
        self.cfg = cfg
        self.E = collector.E
        self.layout = cfg.layout

        self.league = League(agent, cfg.league,
                             agent_factory or self._default_agent_factory)
        self.curriculum = SpawnCurriculum(cfg.env.curriculum)
        self.spawn_rng = np.random.default_rng()

        self.episode_count = 0
        self.env_steps = 0
        self.recent: list[float] = []       # main's last 100 scores, any opponent
        # Draw fraction over recent episodes. With draws scored by the atoms a
        # winrate no longer says how many games ENDED in a draw, and the two
        # numbers move for different reasons — so report it explicitly rather
        # than leaving the regime invisible.
        self.draw_recent: list[float] = []
        self.perf = {"wait": 0.0, "infer": 0.0, "train": 0.0, "cycles": 0}

        H = cfg.net.gru_hidden
        # Capacity: the trigger row count plus margin for the cycle that crosses it.
        self.buf = RolloutBuffer(cfg.ppo.rollout_steps // self.E + 3, self.E,
                                 agent.state_dim, H)
        self.hold = (HoldTracker(self.E, cfg.figar.durations),
                     HoldTracker(self.E, cfg.figar.durations))

        # Recurrent state carried ACROSS decisions and rollouts; reset only at
        # episode ends and phase switches.
        self.h_a = np.zeros((self.E, H), np.float32)
        self.h_c = np.zeros((self.E, H), np.float32)
        self.h_opp = np.zeros((self.E, H), np.float32)

        self.opp: list[tuple[str, int | None]] = [("current", None)] * self.E
        self.cur_mask = np.zeros(self.E, bool)          # opponent is the live model
        self.staller_mask = np.zeros(self.E, bool)      # opponent is a staller
        # This episode's opponent plays argmax instead of sampling.
        self.opp_greedy = np.zeros(self.E, bool)
        # Updates run so far in the CURRENT exploiter phase; gates the
        # `exploiter.actor_frozen` critic warm-up. Reset by `_reset_collection`,
        # which every phase transition already goes through.
        self._exp_updates = 0
        for i in range(self.E):
            self._select_opponent(i)

        # ── value-swing archive (ppo9.archive; see its module docstring) ────
        # Entirely inert (None) unless enabled AND at least one env is
        # tracked — detecting a swing in an untracked env can never produce a
        # restart state, so there is nothing useful to track for it (the
        # correctness-review point about not wasting per-step bookkeeping).
        ac = cfg.archive
        self.archive: Archive | None = None
        self.value_tracker: ValueTracker | None = None
        self._archive_tracked: np.ndarray | None = None
        self._archive_pending: tuple[np.ndarray, np.ndarray] | None = None  # (obs,val) from _infer
        self._archive_req_ids = itertools.count()
        # request_id -> everything needed to finish building the ArchiveEntry
        # once the worker's (best-effort, may never arrive) reply shows up.
        self._archive_open_requests: dict[int, dict[str, Any]] = {}
        # env_idx -> (h_a_seed, h_c_seed), queued by `_archive_seed_hidden`
        # during `_finish_episode` and consumed by `_ingest` once, right
        # after the normal all-envs zero-reset (so it overrides that env's
        # zero rather than being clobbered by it).
        self._archive_hidden_seeds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Per-TRACKED-env detection cooldown, in cycles. A detected swing sits
        # in the retained window for as long as it takes to slide out of it, so
        # without this the SAME swing is re-detected and re-requested EVERY
        # cycle. That is not merely wasteful: it starves the mechanism dead.
        # Measured at 64 tracked envs, it produced ~93 requests/cycle against a
        # 64/cycle reply drain, so `_archive_open_requests` (bounded by
        # `capacity`) evicted 99% of requests before their reply arrived and
        # 99% of arriving replies found no request to match -- insertions
        # stopped permanently after the first few hundred cycles. Waiting a
        # full `history` cycles guarantees the window has completely turned
        # over, so a re-detection is necessarily a genuinely NEW swing rather
        # than the same one seen again. Derived, deliberately not a new knob.
        self._archive_cooldown: np.ndarray | None = None
        # Per-env provenance of the episode CURRENTLY RUNNING: the archive
        # entry it was started from, or NO_ARCHIVE.
        #
        # This exists because `done[i, DONE_ARCHIVE]` describes the reset, i.e.
        # the episode that just STARTED — not the one that just ENDED. Hidden
        # seeding wants the former (it seeds the new episode); the statistics
        # exclusion wants the LATTER, and reading the flag directly there
        # excluded the wrong episode in both directions: a normal episode whose
        # successor happened to be archive-seeded was dropped from the stats,
        # while an archive-started episode whose successor was a normal spawn
        # was counted. Carried across phase transitions on purpose — an
        # archive-started episode still in flight when an exploiter graduates
        # ends during the MAIN phase, and must be excluded there too.
        self._archive_started = np.full(self.E, NO_ARCHIVE, dtype=np.float32)
        if ac.enabled:
            track_count = int(round(ac.track_frac * self.E))
            if track_count > 0:
                self._archive_tracked = np.arange(track_count, dtype=np.int64)
                self.value_tracker = ValueTracker(self._archive_tracked, ac.history,
                                                  agent.state_dim)
                self.archive = Archive(ac)
                self._archive_cooldown = np.zeros(track_count, dtype=np.int64)

    def _default_agent_factory(self, **kw: Any) -> PPOAgent:
        """Build an exploiter sharing the main's shapes but its own atoms/LRs."""
        c = self.cfg
        exp = c.league.exploiter
        return PPOAgent(self.agent.state_dim, self.agent.num_actions,
                        net=c.net, ppo=c.ppo, critic=c.critic, figar=c.figar,
                        entropy=c.entropy, device=str(self.agent.device),
                        clip_eps=exp.clip_eps, actor_lr=exp.actor_lr,
                        epochs=exp.epochs, **kw)

    # ── opponents ──────────────────────────────────────────────────────────
    def _active_agent(self) -> PPOAgent:
        """Whoever is training right now: the exploiter during its phase."""
        if self.league.phase == "exploiter" and self.league.exp_agent is not None:
            return self.league.exp_agent            # type: ignore[return-value]
        return self.agent

    def _select_opponent(self, i: int) -> None:
        # Rolled ONCE per episode, not per decision: a coin flipped every cycle
        # would produce an opponent that is neither greedy nor sampled but an
        # incoherent mixture of both, which is not a policy the agent will ever
        # actually face. See `LeagueConfig.opp_greedy_prob`.
        self.opp_greedy[i] = (self.spawn_rng.random()
                              < self.cfg.league.opp_greedy_prob)
        if self.league.phase == "exploiter":
            self.opp[i] = ("frozen", None)          # best-respond to the frozen main
            self.cur_mask[i] = False
            self.staller_mask[i] = False
            return
        kind, idx = self.league.sample_opponent()
        self.opp[i] = (kind, idx)
        self.cur_mask[i] = kind == "current"
        self.staller_mask[i] = self.league.is_staller(kind, idx)

    def _shift_opponent_indices(self, kind: str) -> None:
        """The oldest pool member of `kind` was evicted; in-flight opponents
        keep the same net one index lower."""
        for j in range(self.E):
            if self.opp[j][0] == kind:
                self.opp[j] = (kind, max(0, (self.opp[j][1] or 0) - 1))
                if kind == "exp":
                    self.staller_mask[j] = self.league.is_staller(*self.opp[j])

    def _opponent_groups(self) -> dict[tuple[str, int | None], np.ndarray]:
        """``{(kind, idx): env indices}`` — one batched forward per distinct net."""
        groups: dict[tuple[str, int | None], list[int]] = {}
        for i in range(self.E):
            groups.setdefault(self.opp[i], []).append(i)
        return {k: np.asarray(v) for k, v in groups.items()}

    def learner_steps(self) -> int:
        return self.buf.learner_steps()

    # ── one cycle ──────────────────────────────────────────────────────────
    def cycle(self) -> None:
        """Advance every environment by one decision block."""
        self.coll.spawn_frac.value = self.curriculum.sample(self.env_steps,
                                                            self.spawn_rng)
        if self.value_tracker is not None:
            # Same lock-free ctx.Value pattern spawn_frac already uses. Gates
            # ALL worker-side archive bookkeeping (retention, requests,
            # spawning) to exploiter phase only — a worker has no other way
            # to know which phase is live.
            self.coll.set_archive_active(self.league.phase == "exploiter")
        t0 = time.perf_counter()
        obs, done = self.coll.sync_obs()
        t1 = time.perf_counter()
        self.env_steps += self.E * self.coll.k
        self.league.add_steps(self.E * self.coll.k)

        self._ingest(done)
        self.coll.send_actions(self._infer(obs))

        t2 = time.perf_counter()
        self.perf["wait"] += t1 - t0
        self.perf["infer"] += t2 - t1
        self.perf["cycles"] += 1

    # -- 1. ingest ----------------------------------------------------------
    def _value_tables(self) -> tuple[np.ndarray, np.ndarray | None]:
        """``(values, staller_values)`` for the CURRENT phase, as ``[3]`` arrays.

        These are used both as the reward paid for an outcome and as the atom
        set the value baseline is computed with — one table per matchup, so the
        two can never disagree. (ppo7 kept them apart and they did: an
        exploiter's critic ran on a 2.0 win scale while it was paid 1.0, making
        a certain win score a NEGATIVE advantage.)
        """
        agent = self._active_agent()
        values = np.asarray(agent.atom_vals, np.float32)
        if self.league.phase == "exploiter":
            return values, None             # the opponent is the frozen main
        return values, np.asarray(self.cfg.league.main_vs_staller_atoms, np.float32)

    def _ingest(self, done: np.ndarray) -> None:
        """Close the open row and run per-episode league bookkeeping."""
        ended = done[:, DONE_ENDED] > 0
        if self.buf.p_valid:
            # The outcome class comes straight from the env now, through a
            # single shared enum, rather than being re-derived from death and
            # timeout flags on this side of the boundary.
            outcome = np.where(ended, done[:, DONE_OUTCOME].astype(np.int64),
                               np.int64(Outcome.NONE))
            values, staller_values = self._value_tables()
            self.buf.commit(ended, outcome, self.staller_mask, values, staller_values)

        if (self.value_tracker is not None and self._archive_pending is not None
                and self.league.phase == "exploiter"):
            # The (obs, value) computed by LAST cycle's _infer is what THIS
            # cycle's `done` actually describes — the same one-cycle lag
            # RolloutBuffer's own commit() already has, for the same reason
            # (whether a decision ended its episode is only known one cycle
            # later). Pushing them paired like this is what keeps the archive
            # ring buffer's own `done` column correctly aligned with its `obs`.
            pend_obs, pend_val = self._archive_pending
            self.value_tracker.push(pend_obs, pend_val, ended.astype(np.float32))
            self._archive_detect()
            self._archive_complete_requests()

        for i in np.nonzero(ended)[0]:
            # A phase transition inside this loop discards the buffers; the
            # remaining ended envs still get their league bookkeeping.
            self._finish_episode(int(i), done[i])

        if ended.any():
            # A new episode starts free, with no memory of the last one.
            self.hold[LEARNER].reset(ended)
            self.hold[OPPONENT].reset(ended)
            self.h_a[ended] = 0.0
            self.h_c[ended] = 0.0
            self.h_opp[ended] = 0.0
            if self._archive_hidden_seeds:
                # Override the zero just written above, for archive-started
                # envs whose entry yielded a valid reconstructed seed — MUST
                # run after the blanket reset, not before, or it would be the
                # one getting clobbered instead.
                for i, (h_a, h_c) in self._archive_hidden_seeds.items():
                    self.h_a[i] = h_a
                    self.h_c[i] = h_c
                self._archive_hidden_seeds.clear()

    # -- 2. inference -------------------------------------------------------
    def _seat_states(self, obs: np.ndarray, seat: int,
                     idx: np.ndarray | None = None) -> np.ndarray:
        """``[n, agent_state_dim]`` — the env observation plus this seat's own
        remaining hold. The env emits `state_dim`; this is what makes it
        `agent_state_dim`."""
        rows = obs[:, seat] if idx is None else obs[idx, seat]
        return np.concatenate([np.ascontiguousarray(rows),
                               self.hold[seat].feature(idx)], axis=1)

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        """Step every seat and return ``[E, 2]`` int64 actions.

        Both seats advance their recurrent state EVERY cycle so it stays aligned
        with the per-cycle rollout rows; the FiGAR hold only gates which sampled
        action is applied.
        """
        active = self._active_agent()
        actions = np.zeros((self.E, 2), dtype=np.int64)

        # Seat 0 (trained): record the hidden BEFORE this row's step, so the
        # update can re-forward the sequence from it.
        l_states = self._seat_states(obs, LEARNER)
        self.buf.set_hidden(self.h_a, self.h_c)
        sa, sd, slp, sv, self.h_a, self.h_c = active.act_step(
            l_states, self.h_a, self.h_c)
        applied, free = self.hold[LEARNER].apply(sa, sd)
        actions[:, LEARNER] = applied
        self.buf.set_learner(l_states, applied, sd, free, slp, sv)
        if self.value_tracker is not None and self.league.phase == "exploiter":
            # Consumed one cycle later by _ingest, paired with the done flags
            # that will then be known for this exact row — see _ingest.
            self._archive_pending = (l_states.copy(), sv.copy())

        # Seat 1: every opponent acts through its own carried hidden state,
        # grouped so each distinct net does exactly one forward.
        s1 = self._seat_states(obs, OPPONENT)
        for key, idxs in self._opponent_groups().items():
            if key[0] == "idle":
                actions[idxs, OPPONENT] = 0     # scripted do-nothing dummy
                continue
            net = active.actor if key[0] == "current" else self.league.opponent_net(*key)
            a_op, d_op = self._act_opponent_group(active, net, s1, idxs)
            applied_o, _ = self.hold[OPPONENT].apply(a_op, d_op, idxs)
            actions[idxs, OPPONENT] = applied_o
        return actions

    def _act_opponent_group(self, active: PPOAgent, net: MinGRUNet,
                            s1: np.ndarray, idxs: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
        """One opponent group's step, padded to a fixed shape on non-CPU devices.

        A mirror-match opponent uses the LIVE actor, which sits on the training
        device, and its group size is binomially sampled — a different number of
        envs every cycle. torch-MPS caches a compiled kernel graph per tensor
        shape and never evicts one, so an ever-changing shape here leaks memory
        without bound. Frozen pool nets stay on CPU and are unaffected; the
        frozen main is on-device but is always the whole of E, a size fixed for
        the phase, so it needs no padding either.
        """
        n = len(idxs)
        states, h = s1[idxs], self.h_opp[idxs]
        greedy = self.opp_greedy[idxs]
        if next(net.parameters()).device.type != "cpu":
            pad = -n % self.cfg.ppo.mps_pad
            if pad:
                states = np.concatenate(
                    [states, np.zeros((pad, states.shape[1]), np.float32)])
                h = np.concatenate([h, np.zeros((pad, h.shape[1]), np.float32)])
                # Padded rows are discarded below, but the mask still has to
                # match the batch or the argmax/sample selection misaligns.
                greedy = np.concatenate([greedy, np.zeros(pad, bool)])
        a_op, d_op, h2 = active.act_actions_step(net, states, h, greedy)
        self.h_opp[idxs] = h2[:n]
        return a_op[:n], d_op[:n]

    # -- episode / phase bookkeeping ----------------------------------------
    def _wr_score(self, outcome: int, i: int) -> float:
        """This episode's result on a 0..1 scale, using the payoff table the
        scored agent was ACTUALLY PAID under.

            score = (atom[outcome] - atom[LOSS]) / (atom[WIN] - atom[LOSS])

        The chess 1/0.5/0 convention scores every draw at half a win no matter
        what the agent's reward said about draws, which made two gates measure
        something other than what they gate on:

        * a COMBAT exploiter is paid `atoms=(1,-1,-1)` — a draw is a LOSS to it
          — yet its graduation gate credited a draw with half a win, so it
          could clear `target_wr` partly by stalling, the exact behaviour those
          atoms exist to forbid. Here a draw scores 0 for it.
        * a STALLER is paid `staller_atoms=(1,1,-1)`, where a draw IS a win.
          Here a draw scores 1 for it, so `target_wr` means the same thing for
          both types.
        * the MAIN is paid `(1,-1,-1)`, so a draw scores 0 — a stalled game is
          a failed kill, not half a success. Against a graduated staller it is
          paid `main_vs_staller_atoms=(1.5,0,-1)` instead, where a draw is
          genuinely partial credit, and scores accordingly.

        Draws are 18-25% of episodes at current episode lengths, so this is
        worth 9-13 points on every winrate it touches — see the note on
        `LeagueConfig.trigger_wr`.
        """
        if not self.cfg.league.wr_from_atoms:
            return {int(Outcome.WIN): 1.0, int(Outcome.DRAW): 0.5,
                    int(Outcome.LOSS): 0.0}[outcome]
        if self.league.phase == "exploiter":
            vals = self._active_agent().atom_vals      # the exploiter's own
        elif self.staller_mask[i]:
            vals = self.cfg.league.main_vs_staller_atoms
        else:
            vals = self.agent.atom_vals
        win, loss = float(vals[int(Outcome.WIN)]), float(vals[int(Outcome.LOSS)])
        span = win - loss
        if span <= 0.0:
            return 0.5                      # degenerate table; nothing to rank
        return min(1.0, max(0.0, (float(vals[outcome]) - loss) / span))

    def _finish_episode(self, i: int, dinfo: np.ndarray) -> None:
        """Score one finished episode and advance the league's phase machine."""
        outcome = int(dinfo[DONE_OUTCOME])
        # Two scores, because they answer different questions. `elo` is the
        # chess convention Elo is defined for and must stay 1/0.5/0. `score`
        # is "what fraction of the payoff available to me did I get", read off
        # the SAME atom table this agent was actually paid under — see
        # `_wr_score`.
        elo = {int(Outcome.WIN): 1.0, int(Outcome.DRAW): 0.5,
               int(Outcome.LOSS): 0.0}[outcome]
        score = self._wr_score(outcome, i)
        self.draw_recent.append(1.0 if outcome == int(Outcome.DRAW) else 0.0)
        del self.draw_recent[:-500]
        self.episode_count += 1

        # `dinfo[DONE_ARCHIVE]` describes the episode that just STARTED (the
        # worker writes it at the reset), NOT the one being scored here. The
        # two consumers therefore want different things, and must not share a
        # read: seeding wants the NEW episode, the statistics exclusion wants
        # the one that just ended — see `_archive_started`.
        was_archive = bool(self._archive_started[i] != NO_ARCHIVE)
        self._archive_started[i] = dinfo[DONE_ARCHIVE]

        # The physics restart is not the whole story — the actor and critic
        # need recurrent context consistent with it too, or an (archived
        # physics state, h=0) combination that never occurred is what actually
        # gets played from. Queue a reconstructed seed (see
        # `_archive_seed_hidden`); applied in `_ingest` AFTER the normal
        # zero-reset so it overrides rather than gets overridden by it.
        if self.archive is not None and dinfo[DONE_ARCHIVE] != NO_ARCHIVE:
            self._archive_seed_hidden(i, int(dinfo[DONE_ARCHIVE]))

        if self.league.phase == "exploiter":
            # Seat 0 IS the exploiter here, so `score` is already its result.
            #
            # An episode that started from the archive is NOT a real matchup
            # result — the initial state was hand-picked for being a hard
            # spot to practice, so a win/loss there says nothing about how the
            # exploiter does against the frozen main from a NORMAL start. It
            # must therefore be excluded from every statistic that judges the
            # exploiter: `record_exp_result` is the ONLY call in this branch
            # that feeds one. Skipping it also skips `phase_eps`/`exp_recent`,
            # so it contributes to neither the graduation gate nor (once
            # graduated) this exploiter's own PFSP weight in the pool — while
            # still training normally via PPO, which happens unconditionally
            # above via `self.buf.commit`. The exclusion is driven by the flag
            # alone, with no dependence on this trainer's archive being
            # enabled: a `done` row that looks archive-started must never count
            # even in a mismatched setup.
            if was_archive:
                self._select_opponent(i)
                return
            if self.league.record_exp_result(score):
                if self.league.finish_exploiter(self.episode_count):
                    self._shift_opponent_indices("exp")
                self._reset_collection()
            else:
                self._select_opponent(i)
            return

        # An archive-started episode still in flight when the exploiter
        # graduated ends HERE, in the main phase. It is no more a real matchup
        # result than it was a moment ago, so it must not reach Elo/PFSP/the
        # displayed win rate either. Snapshot and exploiter-trigger cadence are
        # episode-count driven, not result driven, so they are left alone.
        if not was_archive:
            self.recent.append(score)
            del self.recent[:-100]
            kind, idx = self.opp[i]
            self.league.record_result(kind, idx, score, elo_score=elo)
        if self.league.maybe_snapshot(self.episode_count):
            self._shift_opponent_indices("snap")

        if self.league.should_start_exploiter():
            self.league.start_exploiter()
            self._reset_collection()
        else:
            self._select_opponent(i)

    def _reset_collection(self) -> None:
        """Phase changed: drop in-flight rows (they belong to the old learner),
        drop the recurrent memory, and re-pick every opponent."""
        self.buf.discard()
        self.h_a[:] = 0.0
        self.h_c[:] = 0.0
        self.h_opp[:] = 0.0
        # A new phase means a new (or no) exploiter, so its actor warm-up
        # restarts from zero.
        self._exp_updates = 0
        for i in range(self.E):
            self._select_opponent(i)
        if self.value_tracker is not None:
            # A value/observation from the OLD phase's critic (a different
            # exploiter entirely, or the main) must never seed detection or a
            # restart under the new one — see the module docstring on why a
            # stored value is meaningless outside the critic version that
            # produced it. The worker's own local physics ring and archive
            # copy clear themselves on the matching `archive_active` edge
            # (bonkenv/collect.py), driven by this same phase transition.
            self.value_tracker.clear()
            self._archive_pending = None
            self._archive_open_requests.clear()
            self._archive_hidden_seeds.clear()
            if self.archive is not None:
                # The ENTRY STORE goes too, matching the worker-side clear on
                # the same edge. Without this the workers start the next
                # exploiter with an empty local archive (so a stale entry can
                # never actually seed a restart) while the main process keeps
                # the old exploiter's entries — which then occupy capacity,
                # inflate the reported `arch=` count, and can reject a NEW
                # exploiter's genuine candidate as a positional "duplicate" of
                # a previous exploiter's. Practice states are per-exploiter.
                self.archive.clear()
            if self._archive_cooldown is not None:
                # The window it was suppressing re-detection of is gone too.
                self._archive_cooldown[:] = 0

    # -- value-swing archive (see ppo9/archive.py's module docstring) -------
    def _archive_detect(self) -> None:
        """Stage-1 filter (cheap, over raw possibly-stale-version values),
        then Stage-2 confirmation (a recurrent-consistent rescore under the
        CURRENT critic, anchored to the last verified episode-boundary reset
        in the window) for every TRACKED env. A confirmed swing issues ONE
        on-demand physics request per candidate row — never a continuous
        push; see `bonkenv.collect`'s module docstring for why.

        An env that just issued requests goes on a `history`-cycle cooldown, so
        one swing is requested ONCE rather than re-requested every cycle for as
        long as it remains inside the retained window (see `_archive_cooldown`).
        """
        assert self.value_tracker is not None and self.archive is not None
        assert self._archive_cooldown is not None
        ac = self.cfg.archive
        agent = self._active_agent()
        np.subtract(self._archive_cooldown, 1, out=self._archive_cooldown,
                    where=self._archive_cooldown > 0)
        for local_i, gi in enumerate(self._archive_tracked):
            if self._archive_cooldown[local_i] > 0:
                continue                # this env's last swing is still in its window
            obs_w, val_w, done_w = self.value_tracker.window(local_i)
            idx = last_reset_index(done_w)
            if idx is None:
                continue                # no verified segment in this window at all
            seg_val = self._archive_trim_terminal(val_w[idx:], done_w[idx:], ac)
            if detect_swing(seg_val, ac) is None:
                continue                # cheap filter: not worth a rescore this cycle

            # `idx` is already a verified reset (just proven by the
            # `last_reset_index` call above, over the FULL window). Slicing
            # to `obs_w[idx:]` strips away the antecedent `done` row that
            # proves it, so re-searching via `reconstruct` here would always
            # fail to find a reset in the slice and return None — use the
            # lower-level primitive that trusts the already-verified anchor.
            values2, _, _ = replay_from_reset(agent, obs_w[idx:])
            seg2 = self._archive_trim_terminal(values2, done_w[idx:], ac)
            swing = detect_swing(seg2, ac)
            if swing is None:
                continue                # Stage 1 was a false positive under the current critic

            crit_row = idx + swing.crit_idx        # index into obs_w/done_w
            # The extremum itself, plus up to `neighbor_radius` states either
            # side of the lookback point — never past the extremum (point of
            # the lookback is to practice the ENTRY, not a state already
            # inside the critical situation). If the lookback would reach
            # before the verified reset (a real, common case: a swing whose
            # critical point sits close to the start of the retained
            # segment), CLAMP to the reset itself rather than dropping the
            # candidate — it is still a real, verified point, just with less
            # lookback than asked for.
            offsets = [0] + [o for r in range(1, ac.neighbor_radius + 1) for o in (r, -r)]
            seen_targets: set[int] = set()
            for off in offsets:
                target = max(idx, min(crit_row, crit_row - ac.lookback + off))
                if target in seen_targets:
                    continue                # clamping can collapse several offsets onto one row
                seen_targets.add(target)
                self._archive_request(int(gi), target, len(obs_w), obs_w, done_w,
                                      idx, swing, target - (crit_row - ac.lookback))
            # Requested — don't look at this env again until its window has
            # fully turned over, or this same swing comes back every cycle.
            self._archive_cooldown[local_i] = ac.history

    @staticmethod
    def _archive_trim_terminal(values: np.ndarray, done: np.ndarray, ac: Any) -> np.ndarray:
        """Drop the last `terminal_exclusion` rows when the segment's own tail
        is a real episode end — a value racing to +-1 right before a win/loss
        is the critic correctly reading the outcome, not a meaningful
        mid-game swing."""
        if ac.terminal_exclusion <= 0 or len(values) <= ac.terminal_exclusion:
            return values
        if done[-1] > 0:
            return values[:-ac.terminal_exclusion]
        return values

    def _archive_request(self, global_env_idx: int, target_row: int, window_len: int,
                         obs_w: np.ndarray, done_w: np.ndarray, seg_start: int,
                         swing: Any, lookback_offset: int) -> None:
        """Issue one on-demand physics request and stash everything needed to
        finish building the entry once (if) the reply arrives."""
        req_id = next(self._archive_req_ids)
        # From the VERIFIED reset to the target row — enough to reconstruct a
        # valid hidden state for this row later (see `reconstruct`), and no
        # more: rows before the reset are provably useless for that purpose.
        context_obs = obs_w[seg_start:target_row + 1].copy()
        context_done = done_w[seg_start:target_row + 1].copy()
        x = float(context_obs[-1, SELF_OFF]) / self.cfg.env.obs.pos_scale
        y = float(context_obs[-1, SELF_OFF + 1]) / self.cfg.env.obs.pos_scale
        self._archive_open_requests[req_id] = {
            "global_env_idx": global_env_idx, "context_obs": context_obs,
            "context_done": context_done, "direction": swing.direction,
            "magnitude": swing.magnitude, "source_step": self.env_steps,
            "lookback_offset": lookback_offset, "x": x, "y": y,
        }
        # Bounded FIFO: a request whose reply never arrives (the worker's own
        # ring had already aged the offset out by the time it was asked)
        # must not leak memory forever.
        if len(self._archive_open_requests) > self.cfg.archive.capacity:
            self._archive_open_requests.pop(next(iter(self._archive_open_requests)))
        decisions_ago = (window_len - 1) - target_row
        self.coll.request_snapshot(global_env_idx, decisions_ago, req_id,
                                   self.cfg.run.envs_per_worker)

    def _archive_complete_requests(self) -> None:
        """Drain whatever (rare, best-effort) physics replies have arrived
        and finish building/inserting/broadcasting the corresponding
        entries. A reply that never comes just means that candidate silently
        never materializes — not an error; see the module docstring."""
        assert self.archive is not None
        for snap in self.coll.drain_snapshots():
            req = self._archive_open_requests.pop(snap.request_id, None)
            if req is None or req["global_env_idx"] != snap.global_env_idx:
                continue          # stale (phase changed since) or a mismatch — discard
            entry = self.archive.add(
                snap.phys, snap.heavy_seen, snap.heavy_seen_ticks,
                req["context_obs"], req["context_done"], req["direction"],
                req["magnitude"], req["global_env_idx"], req["source_step"],
                req["lookback_offset"], req["x"], req["y"])
            if entry is not None:
                self.coll.broadcast_entry(entry.id, entry.phys, entry.heavy_seen,
                                          entry.heavy_seen_ticks)

    def _archive_seed_hidden(self, i: int, entry_id: int) -> None:
        """Reconstruct env `i`'s just-started archive episode's actor/critic
        hidden state from the ENTRY's own stored preceding context, under the
        CURRENT weights — never a hidden captured under whatever weights were
        live when the entry was created (that would be exactly the "stale
        hidden from an old policy version" this whole mechanism exists to
        avoid). Falls back to leaving the normal zero-reset in place — the
        same honest "no guess" rule `_archive_detect` follows — if the
        entry's own context has no verifiable reset in it either, or if the
        entry has since been evicted from the archive (it can still be used
        for a reset — the WORKER kept its own copy — even after the main
        process's copy ages out).

        Deliberately scoped to `h_a`/`h_c` (the LEARNER's own actor/critic)
        only — `h_opp` (the frozen opponent's hidden) is not reconstructed:
        only the learner-seat observation history is retained at all, the
        frozen net's own hidden never affects any gradient, and the request
        was specifically about the trainee's recurrent context.

        Uses `replay_from_reset`, NOT `reconstruct` — `entry.context_obs` was
        already sliced to start exactly at its own verified reset row (see
        `_archive_request`), so the ONE row that would let `reconstruct`
        rediscover that boundary (`done[idx-1]>0`) is the row immediately
        BEFORE the slice, which is not part of it. Re-searching a slice for a
        boundary the slicing itself removed the evidence for would silently
        find nothing and this would never fire.
        """
        assert self.archive is not None
        entry = self.archive.get(entry_id)
        if entry is None:
            return
        if len(entry.context_obs) == 0:
            return
        _, h_a, h_c = replay_from_reset(self._active_agent(), entry.context_obs)
        self._archive_hidden_seeds[i] = (h_a.detach().cpu().numpy(),
                                         h_c.detach().cpu().numpy())

    # -- 3. update ----------------------------------------------------------
    def _critic_out(self, states: np.ndarray, hidden: np.ndarray) -> torch.Tensor:
        """Raw critic logits at `states`, with the matching PRE-STEP hidden.

        This MUST go through `critic.step(x, h)` with the row's own carried
        hidden. Calling the net as `critic(x)` defaults the hidden to ZERO and
        silently evaluates a memoryless critic — measured to differ by up to
        0.04 on a freshly-initialized net, and the gap only grows as the critic
        learns to use its memory.
        """
        agent = self._active_agent()
        x = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32)).to(agent.device)
        h = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32)).to(agent.device)
        with torch.no_grad():
            out, _ = agent.critic.step(x, h)
        return out

    def _tail_dist(self, states: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """``[E, n_atoms]`` — the critic's own detached outcome distribution at
        the pending state, used to bootstrap rows whose episode has not resolved."""
        return torch.softmax(self._critic_out(states, hidden), dim=-1).cpu().numpy()

    def _tail_gamma(self, states: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """``[E, G]`` — the aux gamma heads' own detached predictions at the
        pending state, the same bootstrap role `_tail_dist` plays."""
        agent = self._active_agent()
        x = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32)).to(agent.device)
        h = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32)).to(agent.device)
        with torch.no_grad():
            _, _, trunk = agent.critic.step_trunk(x, h)
            return agent.gamma_head(trunk).cpu().numpy()

    def _value_with(self, states: np.ndarray, hidden: np.ndarray,
                    atoms: np.ndarray) -> np.ndarray:
        """Critic value under an ALTERNATIVE payoff table.

        The critic predicts P(win/draw/loss); the value is that distribution's
        expectation under whichever table applies to this matchup. Re-weighting
        rather than retraining is what lets one critic serve every matchup.
        """
        t = torch.tensor(atoms, dtype=torch.float32,
                         device=self._active_agent().device)
        return (torch.softmax(self._critic_out(states, hidden), -1) @ t).cpu().numpy()

    def _entropy_coef(self) -> float:
        """The action-head coefficient this update runs at."""
        if self.league.phase == "exploiter":
            return exploiter_entropy_coef(
                self.cfg.league.exploiter, self.league.phase_steps,
                self.league.exp_ec_start, self.league.exp_gate_hit_at is not None)
        return entropy_coef_at(self.cfg.entropy, self.league.main_steps)

    def _learner_advantages(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        """``(advantages, returns)`` ``[T, E]`` for the learner seat.

        Every cell is a real transition; a cut-off tail (no done on the last
        row) bootstraps from the in-flight pending value.
        """
        b, cfg = self.buf, self.cfg
        r_used = b.b_r[:T]
        if cfg.shaping.dense_reward:
            from bonkenv import REL_OFF
            r_used = r_used + cfg.shaping.dense_coef * targets.approach_shaping(
                b.b_s, b.b_d, T, self.E, REL_OFF,
                cfg.env.obs.pos_scale, cfg.shaping.d_norm)

        v_col = b.b_v[:T].astype(np.float32).copy()
        last_v = np.where(b.b_d[T - 1] > 0, 0.0, b.p_v).astype(np.float32)

        # Against a STALLER the payoff table differs, so the baseline must be
        # re-weighted to match the reward that was actually paid — otherwise a
        # 0 return against a V expecting -1 reads as a +1 advantage and the
        # agent is rewarded for the draw it was supposed to be pushed out of.
        sm = b.b_staller[:T]
        if cfg.critic.categorical and sm.any():
            atoms = np.asarray(cfg.league.main_vs_staller_atoms, np.float32)
            flat = np.nonzero(sm.reshape(-1))[0]
            D = self.agent.state_dim
            v_col.reshape(-1)[flat] = self._value_with(
                b.b_s[:T].reshape(T * self.E, D)[flat],
                b.b_hc[:T].reshape(T * self.E, -1)[flat], atoms)
            cut = (b.b_d[T - 1] <= 0) & sm[T - 1]
            if cut.any():
                last_v = np.where(cut, self._value_with(b.p_s, b.p_hc, atoms), last_v)

        return targets.gae_columns(r_used, v_col, b.b_d[:T], last_v,
                                   cfg.ppo.gamma, cfg.ppo.gae_lambda)

    def run_update(self) -> tuple[dict[str, float], int]:
        """Build targets from the buffer and run one PPO update."""
        t0 = time.perf_counter()
        b, cfg = self.buf, self.cfg
        T, E, D = b.t, self.E, self.agent.state_dim
        agent = self._active_agent()
        adv, ret = self._learner_advantages(T)

        states = b.b_s[:T].copy()
        actions = b.b_a[:T].copy()
        if cfg.ppo.mirror == "sample":
            # Mirror whole ENV trajectories, never individual rows, so each
            # recurrent sequence stays internally consistent.
            em = np.nonzero(np.random.random(E) < 0.5)[0]
            if len(em):
                states[:, em] = mirror_obs_batch(
                    states[:, em].reshape(-1, D), self.layout).reshape(T, len(em), D)
                actions[:, em] = mirror_action_batch(
                    actions[:, em].reshape(-1)).reshape(T, len(em))

        buf: dict[str, Any] = {
            "states": states, "actions": actions, "durations": b.b_dur[:T],
            "free": b.b_free[:T], "old_logp": b.b_lp[:T], "returns": ret,
            "advantages": adv, "values": b.b_v[:T],
            "dones": b.b_d[:T].astype(np.float32),
            "ha0": b.ha0, "hc0": b.b_hc[0],
        }
        if cfg.critic.categorical:
            buf["outcome_target"] = targets.outcome_targets(
                b.b_d[:T], b.b_oc[:T], self._tail_dist(b.p_s, b.p_hc),
                len(agent.atom_vals))
        if cfg.critic.aux_gamma_coef > 0.0:
            buf["gamma_target"] = targets.multi_gamma_targets(
                b.b_d[:T], b.b_oc[:T], cfg.critic.aux_gamma_values,
                np.asarray(agent.atom_vals, np.float32),
                self._tail_gamma(b.p_s, b.p_hc))

        ec = self._entropy_coef()
        # Actor warm-up: hold a fresh exploiter's actor still for its first
        # `actor_frozen` updates so the critic it inherited from the main can
        # re-fit to this matchup before its advantages steer the policy.
        # Counted in UPDATES of this phase, so it is independent of how the
        # rollout size or worker count happen to be set.
        # STRICTLY ADDITIVE: pass True only inside the window, and None (defer
        # to `ppo.freeze_actor`) outside it — never False. Passing False would
        # let this warm-up UN-freeze an actor the config froze on purpose, which
        # is reachable two ways: `--phase 3 --freeze-actor`, where the league
        # runs exploiters under a globally frozen actor, and
        # `--phase 2 --force-exploiter`, where the phase-2 preset sets
        # freeze_actor=True and the flag starts an exploiter phase anyway.
        frozen = None
        if self.league.phase == "exploiter":
            if self._exp_updates < cfg.league.exploiter.actor_frozen:
                frozen = True
            self._exp_updates += 1
        stats = agent.update(buf, entropy_coef=ec,
                             duration_entropy_coef=duration_entropy_coef(cfg.entropy),
                             freeze_actor=frozen)
        # What ACTUALLY happened, config included — not just the warm-up's vote.
        stats["actor_frozen"] = float(bool(frozen) or cfg.ppo.freeze_actor)

        # decided% and mean episode length: the two diagnostics that catch a
        # stall-collapse (a real drop in decided% alongside rising episode
        # length) before it shows up in the winrate.
        stats["decided_pct"] = float(b.b_free[:T].mean() * 100.0)
        n_eps = float(b.b_d[:T].sum())
        stats["mean_ep_decisions"] = (T * E / n_eps) if n_eps > 0 else float("nan")

        b.reset()                   # pendings stay live; they become row 0
        self.perf["train"] += time.perf_counter() - t0
        return stats, T * E

    # ── reporting / persistence ────────────────────────────────────────────
    def win_rate(self) -> float:
        """The main's rolling score over its last 100 episodes."""
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    def draw_rate(self) -> float:
        """Fraction of recent episodes that ENDED in a draw.

        Not derivable from `win_rate` once draws are scored by atoms, and the
        two move for different reasons — a falling winrate with a rising draw
        rate is a stall, with a flat draw rate it is losing fights.
        """
        return (sum(self.draw_recent) / len(self.draw_recent)) if self.draw_recent else 0.0

    def serialize(self) -> dict[str, Any]:
        return {
            "version": 3,
            "algo": "ppo9",
            "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stateDim": self.agent.state_dim,
            "progress": {
                "episodeCount": self.episode_count,
                "envSteps": self.env_steps,
                "currentRating": self.league.current_rating,
            },
            "agent": self.agent.serialize(),
            **self.league.serialize(),
        }

    def load_state(self, obj: dict[str, Any]) -> None:
        if obj.get("algo") != "ppo9":
            raise ValueError(f"not a ppo9 save (algo={obj.get('algo')!r})")
        if obj.get("stateDim") != self.agent.state_dim:
            raise ValueError(f"stateDim mismatch: checkpoint {obj.get('stateDim')} "
                             f"vs agent {self.agent.state_dim}")
        self.agent.load_state(obj["agent"])
        prog = obj.get("progress", {})
        self.episode_count = int(prog.get("episodeCount", 0))
        self.env_steps = int(prog.get("envSteps", 0))
        self.league.load(obj, self.episode_count)
        # Zeroes `_exp_updates`, which is CORRECT here and needs no restored
        # counterpart: `League.serialize` deliberately does not save an
        # in-flight exploiter, and `League.load` always resumes in the main
        # phase with `exp_agent = None`. The next exploiter is therefore a brand
        # new one with a freshly inherited critic, and has earned a full warm-up.
        self._reset_collection()
