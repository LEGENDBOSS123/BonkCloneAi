"""The Trainer: turns a stream of decision-cycles into PPO updates.

Responsibilities, in the order they happen each cycle:

  1. INGEST   fold the block's rewards into the open rollout row, commit it,
              and run per-episode league bookkeeping (results, snapshots,
              exploiter phase transitions).
  2. INFER    run the policies for both seats and open the next row.
  3. UPDATE   once ROLLOUT_STEPS learner steps have accumulated, build the
              targets and hand them to the agent.

Everything below that line lives elsewhere on purpose: physics and episode
resets in the workers (`collect`), population logic in `league`, the array
bookkeeping in `rollout`, FiGAR hold state in `figar`, and the target maths in
`targets`. What is left here is the wiring plus the two things that genuinely
need all of it at once — the two inference paths and the two update paths.

RECURRENT vs FEEDFORWARD. Both paths exist; `config.RECURRENT` picks one.
  * feedforward (legacy TCN): both seats are collected, so a mirror match
    trains twice as many rows per cycle.
  * recurrent (minGRU): the learner seat ONLY. Seat 1 still acts as a full
    opponent, carrying its own hidden state, but its rows are not replayed --
    replaying them would need a second set of per-env hidden states threaded
    through an opponent that changes identity between episodes. Dropping
    train-both keeps the recurrence tractable and correct.
"""

import time

import numpy as np
import torch

from . import config as C
from . import targets
from .collect import VecCollector
from .figar import HoldTracker
from .league import League
from .mirror import mirror_action_batch, mirror_obs_batch
from .ppo import PPOAgent
from .rollout import RolloutBuffer
from .schedules import entropy_coef_at

LEARNER, OPPONENT = 0, 1        # seat indices


class Trainer:
    def __init__(self, collector: VecCollector, agent: PPOAgent):
        self.coll = collector
        self.agent = agent              # the MAIN agent (always what gets saved)
        self.league = League(agent)
        self.E = collector.E

        self.episode_count = 0
        self.spawn_rng = np.random.default_rng()
        self.env_steps = 0
        self.recent = []                # main's last 100 scores (any opponent)
        self.perf = {"wait": 0.0, "infer": 0.0, "train": 0.0, "cycles": 0}

        self.recurrent = bool(getattr(C, "RECURRENT", False))
        # Capacity = the trigger row count plus margin for the cycle that
        # crosses it.
        self.buf = RolloutBuffer(C.ROLLOUT_STEPS // self.E + 3, self.E,
                                 agent.state_dim, recurrent=self.recurrent,
                                 gru_hidden=C.MINGRU_HIDDEN)
        # One FiGAR hold tracker per seat.
        self.hold = (HoldTracker(self.E), HoldTracker(self.E))

        # minGRU hidden state, carried ACROSS decisions and rollouts, reset only
        # at episode ends and phase switches. The learner's actor+critic hidden
        # are trained through; the opponent seat carries its own actor hidden
        # purely so it acts faithfully.
        if self.recurrent:
            H = C.MINGRU_HIDDEN
            self.h_a = np.zeros((self.E, H), np.float32)
            self.h_c = np.zeros((self.E, H), np.float32)
            self.h_opp = np.zeros((self.E, H), np.float32)

        self.opp = [None] * self.E   # ("current"|"snap"|"exp"|"frozen"|"idle", idx)
        self.cur_mask = np.zeros(self.E, dtype=bool)      # opp[i] is "current"
        self.staller_mask = np.zeros(self.E, dtype=bool)  # opp[i] is a STALLER
        for i in range(self.E):
            self._select_opponent(i)

    # ── opponents ──────────────────────────────────────────────────────────
    def _active_agent(self):
        """The agent currently being trained: the exploiter during its phase,
        otherwise the main."""
        return self.league.exp_agent if self.league.phase == "exploiter" else self.agent

    def _select_opponent(self, i):
        if self.league.phase == "exploiter":
            self.opp[i] = ("frozen", None)   # best-respond to the frozen main
            self.cur_mask[i] = False
            self.staller_mask[i] = False
            return
        kind, idx = self.league.sample_opponent()
        self.opp[i] = (kind, idx)
        # Mirror match: both seats are the live learner -> both train.
        self.cur_mask[i] = kind == "current"
        self.staller_mask[i] = self.league.is_staller(kind, idx)

    def _shift_opponent_indices(self, kind):
        """The oldest (kind) pool member was evicted; in-flight opponents keep
        the same net one index lower."""
        for j in range(self.E):
            if self.opp[j][0] == kind:
                self.opp[j] = (kind, max(0, self.opp[j][1] - 1))
                if kind == "exp":
                    self.staller_mask[j] = self.league.is_staller(*self.opp[j])

    def _opponent_groups(self):
        """{(kind, idx): env indices} — one batched forward per distinct net."""
        groups = {}
        for i in range(self.E):
            groups.setdefault(self.opp[i], []).append(i)
        return {k: np.asarray(v) for k, v in groups.items()}

    def learner_steps(self):
        return self.buf.learner_steps()

    # ── spawn curriculum ───────────────────────────────────────────────────
    def spawn_frac(self):
        """Distance curriculum, SAMPLED per cycle (not annealed to one value).

        frac interpolates disc 1 from SPAWN_CURRICULUM_START_M metres (0.0) to
        its natural spawn (1.0). beta relaxes from high (mass near 0 = close
        starts) toward 1 (uniform over the range) as the episode ceiling opens,
        so every distance is visited throughout and the far spawns are always in
        the tail rather than switched on abruptly.
        """
        if not C.SPAWN_CURRICULUM:
            return 1.0
        # Permanent close floor first, then the ramped range for the rest.
        if self.spawn_rng.random() < C.SPAWN_CLOSE_PROB:
            return float(self.spawn_rng.random() ** 5)     # skewed close/mid
        ceil = min(1.0, self.env_steps / max(1, C.SPAWN_CURRICULUM_STEPS))
        beta = 1.0 + C.SPAWN_SHAPE * (1.0 - ceil)
        return float(self.spawn_rng.random() ** beta)

    # ── one decision-block cycle ───────────────────────────────────────────
    def cycle(self):
        self.coll.spawn_frac.value = self.spawn_frac()
        t0 = time.perf_counter()
        obs, brew, done = self.coll.sync_obs()
        t1 = time.perf_counter()
        self.env_steps += self.E * self.coll.k
        self.league.add_steps(self.E * self.coll.k)

        self._ingest(brew, done)
        actions = (self._infer_recurrent(obs) if self.recurrent
                   else self._infer(obs))
        self.coll.send_actions(actions)

        t2 = time.perf_counter()
        self.perf["wait"] += t1 - t0
        self.perf["infer"] += t2 - t1
        self.perf["cycles"] += 1

    # -- 1. ingest ----------------------------------------------------------
    @staticmethod
    def _drew(ended, done):
        """Rows that ended in a DRAW: the clock ran out, or both discs died."""
        return ended & ((done[:, 3] > 0) | ((done[:, 1] > 0) & (done[:, 2] > 0)))

    def _outcome_classes(self, ended, done):
        """Seat-0's true outcome class per env (0=win, 1=draw, 2=loss; -1 = the
        episode is still running) from the death/timeout flags — the same logic
        as the episode score."""
        oc = np.full(self.E, -1, dtype=np.int64)
        if not ended.any():
            return oc
        drew = self._drew(ended, done)
        won = ended & (done[:, 2] > 0) & ~drew
        oc[won] = 0
        oc[drew] = 1
        oc[ended & ~drew & ~won] = 2
        return oc

    def _rescore_draws(self, t, ended, done):
        """Patch the DRAW reward on the row just committed. Only the LEARNER's
        reward is touched.

        Exploiter phase (seat 0 IS the exploiter): a STALLER exploiter scores a
        draw as a WIN — it is trained to time the frozen main out — while a
        normal exploiter uses EXPLOITER_DRAW_REWARD.

        Main phase: the main's timeout against a graduated staller scores 0
        (neutral) rather than the usual -1; run_update re-weights the value
        baseline to match, or a 0 return against a V expecting -1 would read as
        a +1 advantage and reward the draw.
        """
        if not ended.any():
            return
        drew = self._drew(ended, done)
        if not drew.any():
            return
        if self.league.phase == "exploiter":
            dr = (C.EXPLOITER_STALLER_DRAW_REWARD if self.league.exp_is_staller
                  else C.EXPLOITER_DRAW_REWARD)
            if dr != C.DRAW_REWARD:
                self.buf.b_r[t, drew] = dr
        else:
            stalled = self.staller_mask & drew
            if stalled.any():
                self.buf.b_r[t, stalled] = 0.0

    def _ingest(self, brew, done):
        """Close out the block the workers just simulated: fold its rewards into
        the open row, commit that row, and run per-episode league bookkeeping."""
        ended = done[:, 0] > 0
        if self.buf.p_valid:
            self.buf.add_reward(brew)
            t = self.buf.commit(ended, self._outcome_classes(ended, done),
                                self.staller_mask)
            self._rescore_draws(t, ended, done)
        for i in np.nonzero(ended)[0]:
            # A phase transition inside this loop discards the buffers; the
            # remaining ended envs still get their league bookkeeping.
            self._finish_episode(int(i), done[i])
        if ended.any():
            # A new episode starts free, with no memory of the last one.
            self.hold[LEARNER].reset(ended)
            self.hold[OPPONENT].reset(ended)
            if self.recurrent:
                self.h_a[ended] = 0.0
                self.h_c[ended] = 0.0
                self.h_opp[ended] = 0.0

    # -- 2. inference -------------------------------------------------------
    def _seat_states(self, obs, seat, idx=None):
        """The net's input for one seat: the env observation plus that seat's
        own remaining hold, normalised (0 where free). The env emits STATE_DIM;
        this is what makes it AGENT_STATE_DIM."""
        rows = obs[:, seat] if idx is None else obs[idx, seat]
        return np.concatenate([np.ascontiguousarray(rows),
                               self.hold[seat].feature(idx)], axis=1)

    def _infer(self, obs):
        """Feedforward (TCN) inference. The learner and any mirror-match
        opponents share ONE batched forward through the active agent; frozen
        opponents are grouped so each net does a single forward."""
        active = self._active_agent()
        actions = np.zeros((self.E, 2), dtype=np.int64)
        cur = np.nonzero(self.cur_mask)[0]

        l_states = self._seat_states(obs, LEARNER)
        c_states = self._seat_states(obs, OPPONENT, cur) if len(cur) else None
        both = l_states if c_states is None else np.concatenate([l_states, c_states])
        # Pad the batch height to a multiple of 512: torch-MPS caches a compiled
        # graph per tensor shape and never evicts, so a batch that changes size
        # every cycle leaks memory without bound.
        n_real = both.shape[0]
        pad = -n_real % 512
        if pad:
            both = np.concatenate(
                [both, np.zeros((pad, both.shape[1]), dtype=np.float32)])
        acts, durs, lps, vals = active.act_batch(both)
        acts, durs, lps, vals = (acts[:n_real], durs[:n_real],
                                 lps[:n_real], vals[:n_real])

        # seat 0 (learner): re-decide only where the hold expired.
        sd = durs[:self.E]
        applied, free = self.hold[LEARNER].apply(acts[:self.E], sd)
        actions[:, LEARNER] = applied
        self.buf.set_learner(l_states, applied, sd, free,
                             lps[:self.E], vals[:self.E])

        # seat 1: the mirror-match ("current") subset is collected and trained.
        self.buf.open_opponent(self.cur_mask)
        if len(cur):
            cd = durs[self.E:]
            applied_c, free_c = self.hold[OPPONENT].apply(acts[self.E:], cd, cur)
            actions[cur, OPPONENT] = applied_c
            self.buf.set_opponent(cur, c_states, applied_c, cd, free_c,
                                  lps[self.E:], vals[self.E:])

        # Frozen opponents: one batched forward per distinct net. They also hold
        # their action for a sampled duration (they were trained that way), so
        # only re-query where the hold expired.
        for (kind, idx), idxs in self._opponent_groups().items():
            if kind == "current":
                continue
            if kind == "idle":
                actions[idxs, OPPONENT] = 0     # scripted idle: presses nothing
                continue
            net = self.league.opponent_net(kind, idx)
            holds = self.hold[OPPONENT]
            free1 = holds.free_mask(idxs)
            if free1.any():
                fidx = idxs[free1]
                # A frozen net is only queried where the seat is FREE, so its
                # hold feature is 0 by construction.
                o = np.concatenate([np.ascontiguousarray(obs[fidx, OPPONENT]),
                                    np.zeros((len(fidx), 1), np.float32)], 1)
                a_op, d_op = self.agent.act_actions(net, o)
                holds.latch(fidx, a_op, d_op)
            actions[idxs, OPPONENT] = holds.advance(idxs)
        return actions

    def _infer_recurrent(self, obs):
        """minGRU inference. Both seats step their hidden state EVERY cycle so it
        stays aligned with the per-cycle rollout rows; the FiGAR hold only gates
        which sampled ACTION is applied. Only the learner seat is collected."""
        active = self._active_agent()
        actions = np.zeros((self.E, 2), dtype=np.int64)

        # seat 0 (trained): record the hidden BEFORE this row's step, so the
        # update can re-forward the sequence from it.
        l_states = self._seat_states(obs, LEARNER)
        self.buf.set_hidden(self.h_a, self.h_c)
        sa, sd, slp, sv, self.h_a, self.h_c = active.act_step(
            l_states, self.h_a, self.h_c)
        applied, free = self.hold[LEARNER].apply(sa, sd)
        actions[:, LEARNER] = applied
        self.buf.set_learner(l_states, applied, sd, free, slp, sv)
        self.buf.open_opponent()        # seat 1 is not trained in this mode

        # seat 1: every opponent (including a mirror match) acts through its own
        # carried hidden state, grouped so each distinct net does one forward.
        s1 = self._seat_states(obs, OPPONENT)
        for key, idxs in self._opponent_groups().items():
            kind = key[0]
            if kind == "idle":
                actions[idxs, OPPONENT] = 0
                continue
            net = active.actor if kind == "current" else self.league.opponent_net(*key)
            a_op, d_op = self._act_opponent_group(active, net, s1, idxs)
            applied_o, _ = self.hold[OPPONENT].apply(a_op, d_op, idxs)
            actions[idxs, OPPONENT] = applied_o
        return actions

    def _act_opponent_group(self, active, net, s1, idxs):
        """One opponent group's step (recurrent path), padded to a multiple of
        512 rows when `net` lives on a non-CPU device.

        A mirror-match ("current") opponent uses the LIVE actor, which sits on
        `active.device` (MPS in phase 3) -- and this group's size is
        `~OPP_CURRENT_PROB * E`, a DIFFERENT number of envs every cycle
        (binomial sampling). torch-MPS caches a compiled kernel graph per tensor
        shape and never evicts one, so an ever-changing shape here leaks memory
        without bound -- the same failure `_infer` already guards against for
        its learner/mirror batch. Frozen pool/snapshot/exploiter nets stay on
        CPU by default (`League._freeze`) and are unaffected; `frozen_main`
        (exploiter phase) is on the agent's device too but is always the WHOLE
        E, a size fixed across the phase, so it needs no padding.
        """
        n = len(idxs)
        states, h = s1[idxs], self.h_opp[idxs]
        if next(net.parameters()).device.type != "cpu":
            pad = -n % 512
            if pad:
                states = np.concatenate(
                    [states, np.zeros((pad, states.shape[1]), np.float32)])
                h = np.concatenate([h, np.zeros((pad, h.shape[1]), np.float32)])
        a_op, d_op, h2 = active.act_actions_step(net, states, h)
        self.h_opp[idxs] = h2[:n]
        return a_op[:n], d_op[:n]

    # -- episode / phase bookkeeping ----------------------------------------
    def _finish_episode(self, i, dinfo):
        dead0, dead1, timeout = dinfo[1] > 0, dinfo[2] > 0, dinfo[3] > 0
        if timeout or (dead0 and dead1):
            score = 0.5
        elif dead1:
            score = 1.0
        else:
            score = 0.0
        self.episode_count += 1

        if self.league.phase == "exploiter":
            # Graduation winrate uses the standard (wins + 0.5*draws)/games
            # score, same as everywhere else.
            if self.league.record_exp_result(score):
                if self.league.finish_exploiter(self.episode_count):
                    self._shift_opponent_indices("exp")
                self._reset_collection()
            else:
                self._select_opponent(i)
            return

        # Main phase.
        self.recent.append(score)
        del self.recent[:-100]
        kind, idx = self.opp[i]
        self.league.record_result(kind, idx, score)
        if self.league.maybe_snapshot(self.episode_count):
            self._shift_opponent_indices("snap")

        if self.league.should_start_exploiter():
            self.league.start_exploiter()
            self._reset_collection()
        else:
            self._select_opponent(i)

    def _reset_collection(self):
        """Phase changed: drop all in-flight rows/pendings (they belong to the
        old learner), drop the recurrent memory, and re-pick every opponent."""
        self.buf.discard()
        if self.recurrent:
            self.h_a[:] = 0.0
            self.h_c[:] = 0.0
            self.h_opp[:] = 0.0
        for i in range(self.E):
            self._select_opponent(i)

    # -- 3. update ----------------------------------------------------------
    def _critic_out(self, states, hidden=None):
        """Raw critic logits at `states`.

        In recurrent mode this MUST run through `critic.step(x, h)` with the
        state's own carried hidden -- `hidden` is `RolloutBuffer.b_hc`/`p_hc`,
        the pre-step hidden recorded for that exact row. Calling the net as
        `critic(x)` instead (`MinGRUNet.forward`, which defaults an absent
        hidden to ZERO) silently evaluates a memoryless critic: a real carried
        hidden and an all-zero one gave outputs differing by up to 0.04 on a
        freshly-initialized net in testing, and the gap only grows as the
        critic actually learns to use its memory. Feedforward mode has no
        hidden state at all, so `hidden` is ignored there.
        """
        agent = self._active_agent()
        x = torch.from_numpy(np.ascontiguousarray(
            states, dtype=np.float32)).to(agent.device)
        with torch.no_grad():
            if self.recurrent:
                h = torch.from_numpy(np.ascontiguousarray(
                    hidden, dtype=np.float32)).to(agent.device)
                out, _ = agent.critic.step(x, h)
            else:
                out = agent.critic(x)
            return out

    def _tail_dist(self, states, hidden=None):
        """Critic's own (detached) outcome distribution at `states`, used to
        bootstrap rows whose episode has not resolved in this window. Pass the
        matching pre-step hidden (`hidden`) whenever `states` is recurrent."""
        return torch.softmax(self._critic_out(states, hidden), dim=-1).cpu().numpy()

    def _tail_gamma(self, states, hidden=None):
        """Agent's own (detached) multi-gamma auxiliary predictions [E,G] at
        `states` -- used to bootstrap AUX_GAMMA rows whose episode hasn't
        resolved in this window, the same role `_tail_dist` plays for the
        categorical critic. Recurrent only (AUX_GAMMA is wired into
        update_recurrent alone, matching where RUDDER was wired)."""
        agent = self._active_agent()
        x = torch.from_numpy(np.ascontiguousarray(
            states, dtype=np.float32)).to(agent.device)
        with torch.no_grad():
            h = torch.from_numpy(np.ascontiguousarray(
                hidden, dtype=np.float32)).to(agent.device)
            _, _, trunk = agent.critic.step_trunk(x, h)
            return agent.gamma_head(trunk).cpu().numpy()

    def _value_staller(self, states, hidden=None):
        """Critic value with the DRAW atom re-weighted to 0
        (MAIN_VS_STALLER_ATOMS) — the value baseline that matches a 0-reward
        timeout against a staller. Pass the matching pre-step hidden whenever
        `states` is recurrent (see `_critic_out`)."""
        agent = self._active_agent()
        satoms = torch.tensor(C.MAIN_VS_STALLER_ATOMS, dtype=torch.float32,
                              device=agent.device)
        out = self._critic_out(states, hidden)
        return (torch.softmax(out, -1) @ satoms).cpu().numpy()

    def _entropy_coef(self, agent):
        """The coefficient this update runs at.

        Exploiters keep their own two-stage schedule (they must commit low). In
        the MAIN league phase the adaptive controller holds H at
        ENTROPY_TARGET_H, and `ent_coef` was already set by the previous
        update's `adapt_entropy`, so use it as-is. The controller exists to
        track the league's non-stationarity, so phases 1/2 (curriculum,
        LEAGUE_ENABLED=False) keep the plain scheduled coef and their gates are
        not held back.
        """
        if self.league.phase == "exploiter":
            return self.league.exploiter_entropy_coef()
        if C.LEAGUE_ENABLED and getattr(agent, "ent_adaptive", False):
            return agent.ent_coef
        return entropy_coef_at(self.league.main_steps)

    def _maybe_adapt_entropy(self, agent, stats):
        if (self.league.phase != "exploiter" and C.LEAGUE_ENABLED
                and getattr(agent, "ent_adaptive", False)):
            agent.adapt_entropy(stats["entropy"])

    def _rudder_g(self, agent, T, E):
        """RUDDER's return-predictor g(s) at every row of the learner's
        rollout, [T,E]. One extra no-grad forward through the critic's OWN
        trunk (agent.rudder_head rides on it, zero extra network -- see
        PPOAgent.__init__), using the PRE-UPDATE weights -- the same
        convention GAE's own V(s) already uses (collection-time values, not
        live-updated mid-epoch). Recurrent only."""
        b = self.buf
        dev = agent.device
        St = torch.from_numpy(b.b_s[:T].transpose(1, 0, 2).copy()).to(dev)
        don = torch.from_numpy(b.b_d[:T].astype(np.float32)).to(dev)
        reset = torch.zeros(E, T, device=dev)
        reset[:, 1:] = don.transpose(0, 1)[:, :-1]
        hc0 = torch.from_numpy(b.b_hc[0]).to(dev)
        with torch.no_grad():
            _, _, trunk = agent.critic.forward_seq(St, hc0, reset)
            g = agent.rudder_head(trunk).squeeze(-1)        # [E,T]
        return g.transpose(0, 1).cpu().numpy()               # [T,E]

    def _learner_advantages(self, T, E, D):
        """(advantages, returns) [T, E] for the learner seat.

        Every [t, i] cell is a real transition; a cut-off tail (no done on the
        last row) bootstraps from the in-flight pending value.
        """
        b = self.buf
        last_v = np.where(b.b_d[T - 1] > 0, 0.0, b.p_v).astype(np.float32)
        r_used = b.b_r[:T]
        if C.DENSE_REWARD:
            # Phase-1 warm start: pay for moving CLOSER (zero for standing
            # still), which biases the actor toward the opponent without the
            # "park here and collect" trap of an occupancy reward.
            r_used = r_used + C.DENSE_COEF * targets.approach_shaping(
                b.b_s, b.b_d, T, E)
        if C.RUDDER_COEF > 0.0 and self.recurrent:
            g = self._rudder_g(self._active_agent(), T, E)
            r_used = r_used + C.RUDDER_COEF * targets.rudder_redistribute(g, b.b_d[:T])
        # Against a STALLER, re-weight the value baseline so the DRAW atom is 0,
        # matching the 0 return _rescore_draws wrote.
        v_col = b.b_v[:T].astype(np.float32).copy()
        sm = b.b_staller[:T]
        if C.CRITIC_CATEGORICAL and sm.any():
            fidx = np.nonzero(sm.reshape(-1))[0]
            h_rows = (b.b_hc[:T].reshape(T * E, -1)[fidx]
                     if self.recurrent else None)
            v_col.reshape(-1)[fidx] = self._value_staller(
                b.b_s[:T].reshape(T * E, D)[fidx], h_rows)
            cut_st = (b.b_d[T - 1] <= 0) & b.b_staller[T - 1]
            if cut_st.any():
                h_p = b.p_hc if self.recurrent else None
                last_v = np.where(cut_st,
                                  self._value_staller(b.p_s, h_p), last_v)
        return targets.gae_columns(r_used, v_col, b.b_d[:T], last_v)

    def run_update(self):
        t0 = time.perf_counter()
        T, E, D = self.buf.t, self.E, self.agent.state_dim
        adv, ret = self._learner_advantages(T, E, D)
        agent = self._active_agent()
        ec = self._entropy_coef(agent)
        # The FiGAR duration head's coef is DURATION_ENTROPY_MULT (1.5) x the
        # action coef, in BOTH phases -- the duration collapse is opponent-
        # independent and the head needs the stronger pressure to keep long
        # holds sampled -- floored
        # independently of the action coef's own floor, or an exploiter
        # sub-phase's sharp action-head commitment (ec near ENTROPY_COEF_MIN)
        # also crushes duration exploration to the same degree. Measured: the
        # 32/64-cycle buckets went to literally 0% sampled without this floor
        # (see config.DURATION_ENTROPY_COEF_FLOOR).
        if self.recurrent:
            stats, n_rows = self._update_recurrent(agent, T, E, D, adv, ret, ec)
        else:
            stats, n_rows = self._update_feedforward(agent, T, E, D, adv, ret, ec)
        self._maybe_adapt_entropy(agent, stats)
        # decided% and mean episode length: the SAME two diagnostics the
        # original HCA main-agent test used to catch a stall-collapse (see
        # config.py's HCA comment) -- cheap to compute from what's already in
        # the buffer, so every phase gets them, not just ones running HCA.
        stats["decided_pct"] = float(self.buf.b_free[:T].mean() * 100.0)
        n_episodes = float(self.buf.b_d[:T].sum())
        stats["mean_ep_decisions"] = (T * E / n_episodes) if n_episodes > 0 else float("nan")
        self.buf.reset()                # pendings stay live; they become row 0
        self.perf["train"] += time.perf_counter() - t0
        return stats, n_rows

    def _update_recurrent(self, agent, T, E, D, adv, ret, ec):
        """Truncated-BPTT update over the learner's [T, E] sequences."""
        b = self.buf
        oc3 = None
        if C.CRITIC_CATEGORICAL:
            oc3 = targets.outcome_targets(
                b.b_d[:T], b.b_oc[:T],
                self._tail_dist(b.p_s, b.p_hc)).reshape(T, E, -1)
        states3 = b.b_s[:T].copy()
        actions3 = b.b_a[:T].copy()
        if C.MIRROR_MODE in ("sample", "duplicate"):
            # Mirror whole ENV trajectories, never individual rows, so each GRU
            # sequence stays internally consistent.
            em = np.nonzero(np.random.random(E) < 0.5)[0]
            if len(em):
                states3[:, em] = mirror_obs_batch(
                    states3[:, em].reshape(-1, D)).reshape(T, len(em), D)
                actions3[:, em] = mirror_action_batch(
                    actions3[:, em].reshape(-1)).reshape(T, len(em))
        buf = {
            "states": states3, "actions": actions3, "durations": b.b_dur[:T],
            "free": b.b_free[:T], "old_logp": b.b_lp[:T], "returns": ret,
            "advantages": adv, "outcome_target": oc3,
            "dones": b.b_d[:T].astype(np.float32),
            "ha0": b.b_ha[0], "hc0": b.b_hc[0],
        }
        hca_info = {}
        if getattr(agent, "hca_head", None) is not None:
            # (states3/actions3, not b.b_s/b.b_a) -- post-mirror, matching
            # exactly what the update itself trains on this cycle.
            z, valid = targets.hca_episode_labels(b.b_d[:T], b.b_oc[:T])
            agent.add_hca_rows(states3.reshape(T * E, D), actions3.reshape(-1),
                               z.reshape(-1), valid.reshape(-1))
            hca_info = agent.train_hca()   # {} until HCA_MIN_ROWS is reached
        if C.RUDDER_LOSS_COEF > 0.0:
            # Reuses the SAME backward-propagated labels HCA uses, mapped
            # through this agent's OWN atom values (an exploiter's outcome
            # values differ from the main's) rather than one-hot.
            rz, rvalid = targets.hca_episode_labels(b.b_d[:T], b.b_oc[:T])
            atoms = np.asarray(agent.atom_vals, dtype=np.float32)
            buf["rudder_target"] = atoms[np.clip(rz, 0, len(atoms) - 1)]
            buf["rudder_valid"] = rvalid.astype(np.float32)
        if C.AUX_GAMMA_COEF > 0.0:
            tail_g = self._tail_gamma(b.p_s, b.p_hc)       # [E,G]
            atoms_g = np.asarray(agent.atom_vals, dtype=np.float32)
            buf["gamma_target"] = targets.multi_gamma_targets(
                b.b_d[:T], b.b_oc[:T], C.AUX_GAMMA_VALUES, atoms_g, tail_g)
        stats = agent.update_recurrent(
            buf, entropy_coef=ec,
            duration_entropy_coef=max(C.DURATION_ENTROPY_MULT * ec, C.DURATION_ENTROPY_COEF_FLOOR))
        if hca_info:
            stats["hca_z_acc"] = hca_info["z_acc"]
        return stats, T * E

    def _update_feedforward(self, agent, T, E, D, adv, ret, ec):
        """Flat-row update. Both seats are collected here, so a mirror match
        contributes twice as many rows."""
        b = self.buf
        states = b.b_s[:T].reshape(T * E, D).copy()
        aux_t, aux_m = targets.aux_targets(b.b_s, b.b_d, T, E)
        tt_t = targets.time_targets(b.b_d[:T]) if C.CRITIC_TIME_COEF > 0.0 else None
        oc_t = (targets.outcome_targets(b.b_d[:T], b.b_oc[:T],
                                        self._tail_dist(b.p_s))
                if C.CRITIC_CATEGORICAL else None)
        actions = b.b_a[:T].reshape(-1).copy()
        durations = b.b_dur[:T].reshape(-1).copy()
        free = b.b_free[:T].reshape(-1).copy()
        logps = b.b_lp[:T].reshape(-1).copy()
        values = b.b_v[:T].reshape(-1).copy()
        advs, rets = adv.reshape(-1), ret.reshape(-1)
        ddec = targets.approach_shaping(b.b_s, b.b_d, T, E).reshape(-1)

        # Opponent seat: only cells where the opponent was the live "current"
        # model. Invalid cells produce garbage GAE that never leaks INTO valid
        # cells: a valid segment always ends with d=1 (opponents change only at
        # episode end), which zeroes the recursion before the boundary.
        m = b.b_om[:T]
        if m.any():
            last_vo = np.where(b.b_d[T - 1] > 0, 0.0, b.po_v).astype(np.float32)
            adv_o, ret_o = targets.gae_columns(b.b_or[:T], b.b_ov[:T],
                                               b.b_d[:T], last_vo)
            sel = m.reshape(-1)
            states = np.concatenate([states, b.b_os[:T].reshape(T * E, D)[sel]])
            actions = np.concatenate([actions, b.b_oa[:T].reshape(-1)[sel]])
            durations = np.concatenate([durations, b.b_odur[:T].reshape(-1)[sel]])
            free = np.concatenate([free, b.b_ofree[:T].reshape(-1)[sel]])
            logps = np.concatenate([logps, b.b_olp[:T].reshape(-1)[sel]])
            values = np.concatenate([values, b.b_ov[:T].reshape(-1)[sel]])
            advs = np.concatenate([advs, adv_o.reshape(-1)[sel]])
            rets = np.concatenate([rets, ret_o.reshape(-1)[sel]])
            ddec = np.concatenate([ddec, targets.approach_shaping(
                b.b_os, b.b_d, T, E).reshape(-1)[sel]])
            # Aux targets must follow `states` row for row, and the opponent
            # needs its OWN displacement — reusing the learner's would train the
            # head on the wrong disc.
            at_o, am_o = targets.aux_targets(b.b_os, b.b_d, T, E)
            aux_t = np.concatenate([aux_t, at_o[sel]])
            aux_m = np.concatenate([aux_m, am_o[sel]])
            if tt_t is not None:
                # time-to-resolution is seat-independent: both discs experience
                # the same episode end.
                tt_t = np.concatenate([tt_t, tt_t.reshape(T * E)[sel]])
            if oc_t is not None:
                oc_o = targets.outcome_targets(
                    b.b_d[:T], targets.swap_win_loss(b.b_oc[:T]),
                    self._tail_dist(b.po_s))
                oc_t = np.concatenate([oc_t, oc_o[sel]])

        if C.MIRROR_MODE == "duplicate":
            states = np.concatenate([states, mirror_obs_batch(states)])
            actions = np.concatenate([actions, mirror_action_batch(actions)])
            logps, values = np.tile(logps, 2), np.tile(values, 2)
            advs, rets = np.tile(advs, 2), np.tile(rets, 2)
            ddec = np.tile(ddec, 2)   # distance is mirror-invariant
            # duration index and the free mask are mirror-invariant too.
            durations, free = np.tile(durations, 2), np.tile(free, 2)
            # A mirrored world moves the opposite way in x, so every horizon's
            # dx flips sign; the mask is unchanged.
            aux_mir = aux_t.copy()
            aux_mir[:, :, 0] *= -1.0
            aux_t = np.concatenate([aux_t, aux_mir])
            # np.tile on the 2-D mask would tile along the HORIZON axis.
            aux_m = np.concatenate([aux_m, aux_m])
            if oc_t is not None:
                # Mirroring flips the world in x; it does not change who won.
                oc_t = np.concatenate([oc_t, oc_t])
            if tt_t is not None:
                tt_t = np.concatenate([tt_t, tt_t])
        elif C.MIRROR_MODE == "sample":
            # Mirror a random half in place: symmetry without doubling rows.
            mask = np.random.random(len(states)) < 0.5
            states[mask] = mirror_obs_batch(states[mask])
            actions[mask] = mirror_action_batch(actions[mask])
            aux_t[mask, :, 0] *= -1.0

        rollout = {
            "states": states, "actions": actions, "durations": durations,
            "free": free, "logps": logps, "values": values,
            "outcome_target": oc_t, "time_target": tt_t,
            "aux_target": aux_t, "aux_mask": aux_m,
            "advantages": advs, "ddec": ddec, "returns": rets,
        }
        sw = C.SHAPING_START * max(0.0, 1.0 - self.env_steps
                                   / max(1, C.SHAPING_ANNEAL_STEPS))
        # param-noise (exploiter): adapt sigma on the just-collected states
        # BEFORE the update moves the clean actor, then re-perturb for the next
        # rollout.
        if getattr(agent, "param_noise", False):
            agent.adapt_param_noise(states)
        stats = agent.update(
            rollout, entropy_coef=ec, shaping_w=sw,
            duration_entropy_coef=max(C.DURATION_ENTROPY_MULT * ec, C.DURATION_ENTROPY_COEF_FLOOR))
        if getattr(agent, "param_noise", False):
            agent.resample_param_noise()
        return stats, states.shape[0]

    def win_rate(self):
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    # ── save / load ────────────────────────────────────────────────────────
    def serialize(self):
        return {
            "version": 2,
            "algo": "ppo",
            "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stateDim": self.agent.state_dim,
            "progress": {
                "episodeCount": self.episode_count,
                "envSteps": self.env_steps,
                "currentRating": self.league.current_rating,
            },
            "agent": self.agent.serialize(),
            **self.league.serialize(),      # "snapshots" + "league"
        }

    def load_state(self, obj):
        if obj.get("algo") != "ppo":
            raise ValueError(f"not a ppo save (algo={obj.get('algo')})")
        if obj.get("stateDim") != self.agent.state_dim:
            raise ValueError(f"stateDim mismatch: checkpoint "
                             f"{obj.get('stateDim')} vs agent {self.agent.state_dim}")
        self.agent.load_state(obj["agent"])
        prog = obj.get("progress", {})
        self.episode_count = prog.get("episodeCount", 0)
        self.env_steps = prog.get("envSteps", 0)
        self.league.load(obj, self.episode_count)
        self._reset_collection()
