"""bonk2 trainer: multiprocess self-play PPO with league opponents.

Workers collect (collect.VecCollector), the main process infers, learns, and
does the bookkeeping; all population logic lives in league.League. Run:

  python -m ppo6.train --workers 10 --device mps

Device strategy (measured on Apple Silicon at E=2560, [512,512]): the PPO
update is ~2x faster on MPS and big-batch inference is a wash, so --device mps
puts the trained agents (main / exploiter / frozen_main) there; frozen POOL
nets always stay on CPU, where many small per-net batches beat GPU dispatch
overhead. Plain --device cpu remains fully supported.

Ctrl-C saves a checkpoint and shuts the workers down. Checkpoints are the same
JSON shape as v1 (stateDim: 34), so bonk.export_model and the record-based
loaders (play3.mjs, ppo6.play, ppo6.eval_h2h) work unchanged.
"""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .collect import VecCollector
from .env import mirror_action_batch, mirror_obs_batch
from .league import League
from .ppo import PPOAgent, entropy_coef_at

REPO = Path(__file__).resolve().parents[2]


class Trainer:
    """PPO over a VecCollector: per-env streams/pendings live here; physics and
    episode resets live in the workers; the population lives in the League."""

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

        # Rollout buffers. Every env advances one row per cycle (pendings are
        # completed in lockstep), so streams are flat [T, E] arrays and GAE
        # vectorizes across all envs — per-env Python lists were the main-
        # process bottleneck at E≈2560.
        E, D = self.E, agent.state_dim
        self.T = C.ROLLOUT_STEPS // E + 3          # capacity (trigger + margin)
        self.b_s = np.zeros((self.T, E, D), dtype=np.float32)   # learner seat
        self.b_a = np.zeros((self.T, E), dtype=np.int64)
        self.b_lp = np.zeros((self.T, E), dtype=np.float32)
        self.b_v = np.zeros((self.T, E), dtype=np.float32)
        self.b_r = np.zeros((self.T, E), dtype=np.float32)
        self.b_d = np.zeros((self.T, E), dtype=np.float32)
        # Seat-0 outcome CLASS per terminal row (0=win, 1=draw, 2=loss; -1 =
        # non-terminal). The categorical critic labels from THIS actual result,
        # not from nearest-atom-to-reward -- which silently merges draw and loss
        # whenever their rewards are equal (they are, both -1).
        self.b_oc = np.full((self.T, E), -1, dtype=np.int64)
        self.b_os = np.zeros((self.T, E, D), dtype=np.float32)  # opponent seat
        self.b_oa = np.zeros((self.T, E), dtype=np.int64)       # ("current" envs
        self.b_olp = np.zeros((self.T, E), dtype=np.float32)    #  only, masked
        self.b_ov = np.zeros((self.T, E), dtype=np.float32)     #  by b_om)
        self.b_or = np.zeros((self.T, E), dtype=np.float32)
        self.b_om = np.zeros((self.T, E), dtype=bool)
        self.t = 0

        # In-flight decisions (one per env, arrays across E).
        self.p_valid = False            # no pendings until the first inference
        self.p_s = np.zeros((E, D), dtype=np.float32)
        self.p_a = np.zeros(E, dtype=np.int64)
        self.p_lp = np.zeros(E, dtype=np.float32)
        self.p_v = np.zeros(E, dtype=np.float32)
        self.p_r = np.zeros(E, dtype=np.float32)
        self.po_s = np.zeros((E, D), dtype=np.float32)
        self.po_a = np.zeros(E, dtype=np.int64)
        self.po_lp = np.zeros(E, dtype=np.float32)
        self.po_v = np.zeros(E, dtype=np.float32)
        self.po_r = np.zeros(E, dtype=np.float32)
        self.po_m = np.zeros(E, dtype=bool)        # opponent pend validity

        self.opp = [None] * self.E      # ("current",None)|("snap",i)|("exp",i)|("frozen",None)
        self.cur_mask = np.zeros(E, dtype=bool)    # opp[i] is "current"
        for i in range(self.E):
            self._select_opponent(i)

    def _active_agent(self):
        return self.league.exp_agent if self.league.phase == "exploiter" else self.agent

    def _select_opponent(self, i):
        if self.league.phase == "exploiter":
            self.opp[i] = ("frozen", None)   # best-respond to the frozen main
            self.cur_mask[i] = False
        else:
            kind, idx = self.league.sample_opponent()
            self.opp[i] = (kind, idx)
            # Mirror match: both seats are the live learner -> both train.
            self.cur_mask[i] = kind == "current"

    def _shift_opponent_indices(self, kind):
        """The oldest (kind) pool member was evicted; in-flight opponents keep
        the same net one index lower."""
        for j in range(self.E):
            if self.opp[j][0] == kind:
                self.opp[j] = (kind, max(0, self.opp[j][1] - 1))

    def learner_steps(self):
        return self.t * self.E

    # ── one decision-block cycle ───────────────────────────────────────────────
    def spawn_frac(self):
        """Distance curriculum, SAMPLED per cycle (not annealed to one value).

        frac interpolates disc 1 from SPAWN_CURRICULUM_START_M metres (0.0) to
        its natural spawn (1.0). beta relaxes from high (mass near 0 = close
        starts) toward 1 (uniform over the range) as the episode ceiling opens,
        so every distance is visited throughout and the far spawns are always
        in the tail rather than switched on abruptly.
        """
        if not C.SPAWN_CURRICULUM:
            return 1.0
        # Permanent close floor first, then the ramped range for the rest.
        if self.spawn_rng.random() < C.SPAWN_CLOSE_PROB:
            return float(self.spawn_rng.random() ** 5)     # skewed close/mid
        ceil = min(1.0, self.env_steps / max(1, C.SPAWN_CURRICULUM_STEPS))
        beta = 1.0 + C.SPAWN_SHAPE * (1.0 - ceil)
        return float(self.spawn_rng.random() ** beta)

    def cycle(self):
        self.coll.spawn_frac.value = self.spawn_frac()
        t0 = time.perf_counter()
        obs, brew, done = self.coll.sync_obs()
        t1 = time.perf_counter()
        self.env_steps += self.E * self.coll.k
        self.league.add_steps(self.E * self.coll.k)

        # 1. Fold the finished block into pendings and commit them as row t.
        ended = done[:, 0] > 0
        if self.p_valid:
            t = self.t
            self.p_r += brew[:, 0]
            self.b_s[t] = self.p_s
            self.b_a[t] = self.p_a
            self.b_lp[t] = self.p_lp
            self.b_v[t] = self.p_v
            self.b_r[t] = self.p_r
            # Exploiter phase: seat 0 IS the exploiter, and a DRAW is worth
            # EXPLOITER_DRAW_REWARD (0) to it rather than the main's -1 --
            # holding the main to a draw is a partial success. Nearest-atom
            # against EXPLOITER_ATOMS then labels it the draw class. Only the
            # LEARNER's reward is touched; the frozen main is not training.
            if (self.league.phase == "exploiter" and ended.any()
                    and C.EXPLOITER_DRAW_REWARD != C.DRAW_REWARD):
                drew = ended & ((done[:, 3] > 0)
                                | ((done[:, 1] > 0) & (done[:, 2] > 0)))
                if drew.any():
                    self.b_r[t, drew] = C.EXPLOITER_DRAW_REWARD
            self.b_d[t] = ended
            # True seat-0 outcome class from the death/timeout flags (same logic
            # as _finish_episode's score): draw = timeout or both dead; win = the
            # opponent (seat 1) died; loss = seat 0 died.
            oc = np.full(self.E, -1, dtype=np.int64)
            if ended.any():
                drew = ended & ((done[:, 3] > 0)
                                | ((done[:, 1] > 0) & (done[:, 2] > 0)))
                won = ended & (done[:, 2] > 0) & ~drew
                lost = ended & ~drew & ~won
                oc[won] = 0
                oc[drew] = 1
                oc[lost] = 2
            self.b_oc[t] = oc
            self.po_r += brew[:, 1]
            self.b_os[t] = self.po_s
            self.b_oa[t] = self.po_a
            self.b_olp[t] = self.po_lp
            self.b_ov[t] = self.po_v
            self.b_or[t] = self.po_r
            self.b_om[t] = self.po_m
            self.t = t + 1
            self.p_valid = False
        for i in np.nonzero(ended)[0]:
            self._finish_episode(int(i), done[i])
            # A phase transition inside this loop resets the buffers; the
            # remaining ended envs still get their league bookkeeping.

        # 2. Batched inference. The learner seat uses the ACTIVE agent (main, or
        #    the exploiter during its phase); opponents route by kind.
        active = self._active_agent()
        actions = np.zeros((self.E, 2), dtype=np.int64)
        # Learner seats + "current" opponent seats share the same net in main
        # phase (and cur is empty in exploiter phase), so run them as ONE batch.
        cur = np.nonzero(self.cur_mask)[0]
        l_states = np.ascontiguousarray(obs[:, 0])
        both = (np.concatenate([l_states, obs[cur, 1]]) if len(cur) else l_states)
        # Pad the batch height to a multiple of 512: torch-MPS compiles and
        # caches a kernel graph PER TENSOR SHAPE and never evicts, so feeding
        # it a different height every cycle leaks ~MB per new shape (observed
        # 60 GB footprint over hours). Bucketing keeps the shape set tiny.
        n_real = both.shape[0]
        pad = -n_real % 512
        if pad:
            both = np.concatenate(
                [both, np.zeros((pad, both.shape[1]), dtype=np.float32)])
        acts, lps, vals = active.act_batch(both)
        acts, lps, vals = acts[:n_real], lps[:n_real], vals[:n_real]
        actions[:, 0] = acts[:self.E]
        self.p_s[:] = l_states
        self.p_a[:] = acts[:self.E]
        self.p_lp[:] = lps[:self.E]
        self.p_v[:] = vals[:self.E]
        self.p_r[:] = 0.0
        self.po_m[:] = self.cur_mask
        self.po_r[:] = 0.0
        if len(cur):
            actions[cur, 1] = acts[self.E:]
            self.po_s[cur] = both[self.E:n_real]   # exclude padding rows
            self.po_a[cur] = acts[self.E:]
            self.po_lp[cur] = lps[self.E:]
            self.po_v[cur] = vals[self.E:]
        self.p_valid = True
        # Frozen opponents grouped so each net does one batched forward.
        groups = {}
        for i in range(self.E):
            kind, idx = self.opp[i]
            if kind != "current":
                groups.setdefault((kind, idx), []).append(i)
        for (kind, idx), idxs in groups.items():
            if kind == "idle":
                actions[idxs, 1] = 0        # scripted idle: presses nothing
                continue
            net = self.league.opponent_net(kind, idx)
            acts = self.agent.act_actions(net, np.ascontiguousarray(obs[idxs, 1]))
            actions[idxs, 1] = acts

        self.coll.send_actions(actions)
        t2 = time.perf_counter()
        self.perf["wait"] += t1 - t0
        self.perf["infer"] += t2 - t1
        self.perf["cycles"] += 1

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
        old learner) and re-pick every opponent."""
        self.t = 0
        self.p_valid = False
        self.po_m[:] = False
        for i in range(self.E):
            self._select_opponent(i)

    # ── PPO update ─────────────────────────────────────────────────────────────
    @staticmethod
    def _gae_columns(r, v, d, last_v):
        """GAE over [T, E] arrays, vectorized across the E columns."""
        T = r.shape[0]
        adv = np.zeros_like(r)
        gae = np.zeros(r.shape[1], dtype=np.float32)
        for t in range(T - 1, -1, -1):
            nt = 1.0 - d[t]
            next_v = last_v if t == T - 1 else v[t + 1]
            delta = r[t] + C.GAMMA * next_v * nt - v[t]
            gae = delta + C.GAMMA * C.GAE_LAMBDA * nt * gae
            adv[t] = gae
        return adv, adv + v

    @staticmethod
    def _aux_targets(buf_s, buf_d, T, E):
        """(target [T*E, H, 2], mask [T*E, H]) — own displacement h steps ahead.

        obs[0:2] is the agent's own position, already POS_SCALEd, so a plain
        difference is the displacement in the same units the net sees. A
        horizon is masked off when the episode ends before it is reached: those
        rows would otherwise be trained against the NEXT episode's spawn, which
        is the stale-target bug that silently corrupted bonk4's aux data.
        """
        H = C.AUX_POS_HORIZONS
        tgt = np.zeros((T, E, len(H), 2), dtype=np.float32)
        msk = np.zeros((T, E, len(H)), dtype=bool)
        # ended[t] = an episode boundary occurred at or before t, per env
        for hi, h in enumerate(H):
            if T - h <= 0:
                continue
            t0 = np.arange(T - h)
            tgt[t0, :, hi, :] = buf_s[t0 + h, :, 0:2] - buf_s[t0, :, 0:2]
            # valid only if no done in (t, t+h]
            crossed = np.zeros((T - h, E), dtype=bool)
            for k in range(h):
                crossed |= buf_d[t0 + k] > 0
            msk[t0, :, hi] = ~crossed
        return tgt.reshape(T * E, len(H), 2), msk.reshape(T * E, len(H))

    def _time_targets(self, T, E):
        """[T*E] decisions remaining until the episode resolves, normalised.

        Propagated BACKWARD from each episode end, exactly like the outcome
        target. Rows in a window that never resolves are censored: they are
        given the largest value seen so far and masked out is not possible with
        an MSE head, so instead they carry the distance to the window edge,
        which is a LOWER BOUND on the true remaining time. That biases the head
        toward under-estimating on unresolved tails -- acceptable for an
        auxiliary signal, and the alternative (dropping them) would train the
        head only on short episodes.
        """
        cap = max(1.0, C.MAX_EPISODE_STEPS / C.ACTION_REPEAT)
        tgt = np.zeros((T, E), dtype=np.float32)
        since = np.zeros(E, dtype=np.float32)   # decisions until resolution
        for t in range(T - 1, -1, -1):
            d = self.b_d[t] > 0
            since = since + 1.0
            since[d] = 1.0                       # resolves at this row
            tgt[t] = since
        return (tgt / cap).reshape(T * E)

    def _tail_dist(self, states):
        """Critic's own (detached) outcome distribution at the pending state,
        used to bootstrap rows whose episode has not resolved in this window."""
        agent = self._active_agent()
        with torch.no_grad():
            out = agent.critic(torch.from_numpy(
                np.ascontiguousarray(states)).to(agent.device))
            return torch.softmax(out, dim=-1).cpu().numpy()

    def _outcome_targets(self, T, E, tail_dist, outcome=None):
        """[T*E, 3] target distribution over the terminal outcome.

        With gamma = 1 and terminal-only rewards the distributional Bellman
        backup is Z(s_t) = Z(s_{t+1}) away from terminal, so the target simply
        propagates BACKWARD from each episode end. Rows whose episode has not
        resolved inside the window inherit `tail_dist`, the critic's own
        (detached) prediction at the pending state -- which is what keeps this
        compatible with bootstrapped windows instead of forcing the trainer to
        wait for whole episodes, as ppo4 had to.

        The realised class is the ACTUAL outcome (b_oc: 0=win/1=draw/2=loss),
        NOT nearest-atom-to-reward -- the old shortcut merged draw and loss
        whenever their rewards were equal (they are, both -1), so the critic
        never learned "loss" and read every lost position as a draw. `outcome`
        selects WHOSE result: the opponent seat passes the win/loss-swapped
        array, or its rows would carry the learner's result.
        """
        outcome = self.b_oc if outcome is None else outcome
        n = len(C.CRITIC_ATOMS)
        tgt = np.zeros((T, E, n), dtype=np.float32)
        cur = tail_dist.astype(np.float32).copy()
        for t in range(T - 1, -1, -1):
            d = self.b_d[t] > 0
            if d.any():
                cls = outcome[t][d]
                oh = np.zeros((int(d.sum()), n), dtype=np.float32)
                oh[np.arange(len(cls)), cls] = 1.0
                cur = cur.copy()
                cur[d] = oh
            tgt[t] = cur
        return tgt.reshape(T * E, n)

    @staticmethod
    def _closeness(buf_s, T, E):
        """[T,E] RAW per-step closeness reward: Phi = 1/(1 + dist/D_NORM), a
        bounded 1/x function of distance to the opponent (1 at contact -> 0 far;
        rel dx,dy at obs 24,25, POS_SCALEd).

        Unlike a potential DIFFERENCE this is an occupancy reward paid every step
        for BEING close, so it is deliberately NOT policy-invariant -- it biases
        the actor toward seeking the opponent. That is intentional: phase 1 only
        builds a warm-start APPROACH PRIOR that phase 2/3 refine, so shifting the
        optimum toward "go to the opponent" is the point. It is still safe
        against suicide because dying ends the episode (and the reward stream),
        so staying alive and close beats a brief lunge into a pit.
        """
        rel = buf_s[:T, :, 24:26]
        dist = np.hypot(rel[:, :, 0], rel[:, :, 1]) / C.POS_SCALE
        return (1.0 / (1.0 + dist / C.SHAPING_D_NORM)).astype(np.float32)

    @staticmethod
    def _approach_shaping(buf_s, buf_d, T, E):
        """[T,E] APPROACH reward: the rise in the bounded 1/x potential
        Phi = 1/(1 + dist/D_NORM) from this decision to the next (rel dx,dy at
        obs 24,25, POS_SCALEd; 1 at contact -> 0 far).

        Rewards the CHANGE, not the level: getting closer pays, standing still
        pays ZERO, backing off pays negative. That is the key difference from an
        occupancy reward -- with occupancy the agent just parks in a decent spot
        and collects it (measured: it stands still), because moving on fragile
        terrain risks a fatal fall. Here there is no safe standing income, so the
        only way to earn is to move toward the opponent. Zeroed at terminal rows
        (no clawback): deliberately NOT policy-invariant, so it biases the phase-1
        warm-start actor toward approach.
        """
        rel = buf_s[:T, :, 24:26]
        dist = np.hypot(rel[:, :, 0], rel[:, :, 1]) / C.POS_SCALE
        phi = 1.0 / (1.0 + dist / C.SHAPING_D_NORM)    # bounded (0,1], closer=higher
        dd = np.zeros((T, E), dtype=np.float32)
        dd[:T - 1] = phi[1:T] - phi[:T - 1]            # positive = approached
        dd[buf_d[:T] > 0] = 0.0                         # not across resets
        return dd

    def run_update(self):
        t0 = time.perf_counter()
        T, E, D = self.t, self.E, self.agent.state_dim

        # Learner seat: every [t, i] cell is a real transition. Cut-off tails
        # (d[T-1]=0) bootstrap from the in-flight pending value.
        last_v = np.where(self.b_d[T - 1] > 0, 0.0, self.p_v).astype(np.float32)
        r_used = self.b_r[:T]
        if C.DENSE_REWARD:
            # APPROACH reward in the RETURN (phase-1 warm-start): pays for moving
            # closer, zero for standing still -> biases the actor toward the
            # opponent without an occupancy "park here" trap. Scalar critic
            # baselines it via GAE.
            r_used = r_used + C.DENSE_COEF * self._approach_shaping(self.b_s, self.b_d, T, E)
        adv, ret = self._gae_columns(r_used, self.b_v[:T], self.b_d[:T], last_v)
        states = self.b_s[:T].reshape(T * E, D).copy()
        aux_t, aux_m = self._aux_targets(self.b_s, self.b_d, T, E)
        tt_t = self._time_targets(T, E) if C.CRITIC_TIME_COEF > 0.0 else None
        oc_t = None
        if C.CRITIC_CATEGORICAL:
            oc_t = self._outcome_targets(T, E, self._tail_dist(self.p_s))
        actions = self.b_a[:T].reshape(-1).copy()
        logps = self.b_lp[:T].reshape(-1).copy()
        values = self.b_v[:T].reshape(-1).copy()
        advs, rets = adv.reshape(-1), ret.reshape(-1)
        ddec = self._approach_shaping(self.b_s, self.b_d, T, E).reshape(-1)

        # Opponent seat: only cells where the opponent was "current" (b_om).
        # Invalid cells produce garbage GAE that never leaks INTO valid cells:
        # a valid segment always ends with d=1 (opponents change only at
        # episode end), which zeroes the recursion before the boundary.
        m = self.b_om[:T]
        if m.any():
            last_vo = np.where(self.b_d[T - 1] > 0, 0.0, self.po_v).astype(np.float32)
            adv_o, ret_o = self._gae_columns(self.b_or[:T], self.b_ov[:T],
                                             self.b_d[:T], last_vo)
            sel = m.reshape(-1)
            states = np.concatenate([states, self.b_os[:T].reshape(T * E, D)[sel]])
            actions = np.concatenate([actions, self.b_oa[:T].reshape(-1)[sel]])
            logps = np.concatenate([logps, self.b_olp[:T].reshape(-1)[sel]])
            values = np.concatenate([values, self.b_ov[:T].reshape(-1)[sel]])
            advs = np.concatenate([advs, adv_o.reshape(-1)[sel]])
            rets = np.concatenate([rets, ret_o.reshape(-1)[sel]])
            ddec = np.concatenate([ddec,
                self._approach_shaping(self.b_os, self.b_d, T, E).reshape(-1)[sel]])
            # Aux targets must follow `states` row for row. The opponent seat
            # needs its OWN displacement, computed from its own observation
            # block -- reusing the learner's would train the head on the wrong
            # disc.
            at_o, am_o = self._aux_targets(self.b_os, self.b_d, T, E)
            aux_t = np.concatenate([aux_t, at_o[sel]])
            aux_m = np.concatenate([aux_m, am_o[sel]])
            if tt_t is not None:
                # time-to-resolution is seat-independent: both discs
                # experience the same episode end.
                tt_t = np.concatenate([tt_t, tt_t.reshape(T*E)[sel]])
            if oc_t is not None:
                # Opponent's outcome = seat-0's with win<->loss swapped (0<->2);
                # draw (1) and non-terminal (-1) unchanged. A learner win is an
                # opponent loss.
                opp_oc = np.where(self.b_oc == 0, 2,
                                  np.where(self.b_oc == 2, 0, self.b_oc))
                oc_o = self._outcome_targets(T, E, self._tail_dist(self.po_s),
                                             outcome=opp_oc)
                oc_t = np.concatenate([oc_t, oc_o[sel]])
        if C.MIRROR_MODE == "duplicate":
            states = np.concatenate([states, mirror_obs_batch(states)])
            actions = np.concatenate([actions, mirror_action_batch(actions)])
            logps, values = np.tile(logps, 2), np.tile(values, 2)
            advs, rets = np.tile(advs, 2), np.tile(rets, 2)
            ddec = np.tile(ddec, 2)   # distance is mirror-invariant
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
            "states": states,
            "actions": actions,
            "logps": logps,
            "values": values,
            "outcome_target": oc_t,
            "time_target": tt_t,
            "aux_target": aux_t,
            "aux_mask": aux_m,
            "advantages": advs,
            "ddec": ddec,
            "returns": rets,
        }
        if self.league.phase == "exploiter":
            ec = self.league.exploiter_entropy_coef()
        else:
            ec = entropy_coef_at(self.league.main_steps)
        sw = C.SHAPING_START * max(0.0, 1.0 - self.env_steps
                                  / max(1, C.SHAPING_ANNEAL_STEPS))
        stats = self._active_agent().update(rollout, entropy_coef=ec,
                                            shaping_w=sw)
        self.t = 0                      # restart the buffers; pendings stay live
        self.perf["train"] += time.perf_counter() - t0
        return stats, rollout["states"].shape[0]

    def win_rate(self):
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    # ── save / load ────────────────────────────────────────────────────────────
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"))
    ap.add_argument("--load")
    ap.add_argument("--warm-actor",
                    help="load ONLY the agent weights from this checkpoint "
                         "(fresh league, progress reset) -- for a map switch")
    ap.add_argument("--out", default=str(REPO / "runs/ppo2"))
    ap.add_argument("--steps", type=int, default=0,
                    help="stop after this many ENV STEPS (primary unit)")
    ap.add_argument("--episodes", type=int, default=100_000_000,
                    help="secondary cap; --steps takes precedence when set")
    ap.add_argument("--save-every", type=int, default=400_000,
                    help="checkpoint interval in ENV STEPS")
    ap.add_argument("--replay-every", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs-per-worker", type=int, default=320)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scalar-critic", action="store_true",
                    help="phase 1: scalar value critic instead of categorical")
    ap.add_argument("--dense-reward", action="store_true",
                    help="phase 1: dense distance-decrease reward in the return")
    ap.add_argument("--dense-coef", type=float, default=None)
    ap.add_argument("--all-idle", action="store_true",
                    help="phase 1: opponent is the stationary bot 100% of the time")
    ap.add_argument("--freeze-actor", action="store_true",
                    help="phase 2: train only the critic, actor held fixed")
    ap.add_argument("--no-league", action="store_true",
                    help="disable snapshots + exploiters (implied by --all-idle)")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="cap torch CPU threads (0 = leave default)")
    ap.add_argument("--exploiter-trigger-wr", type=float,
                    help="override config.EXPLOITER_TRIGGER_WR")
    ap.add_argument("--exploiter-max-interval", type=int,
                    help="override config.EXPLOITER_MAX_INTERVAL")
    ap.add_argument("--exploiter-max-episodes", type=int,
                    help="override config.EXPLOITER_MAX_EPISODES")
    args = ap.parse_args()
    if args.scalar_critic: C.CRITIC_CATEGORICAL = False
    if args.dense_reward:  C.DENSE_REWARD = True
    if args.dense_coef is not None: C.DENSE_COEF = args.dense_coef
    if args.all_idle:      C.OPP_IDLE_PROB = 1.0
    if args.freeze_actor:  C.FREEZE_ACTOR = True
    if args.no_league or args.all_idle: C.LEAGUE_ENABLED = False
    print(f"  phase flags: scalar_critic={not C.CRITIC_CATEGORICAL} "
          f"dense={C.DENSE_REWARD}({C.DENSE_COEF}) all_idle={C.OPP_IDLE_PROB==1.0} "
          f"freeze_actor={C.FREEZE_ACTOR} league={C.LEAGUE_ENABLED}")

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    for arg, name in ((args.exploiter_trigger_wr, "EXPLOITER_TRIGGER_WR"),
                      (args.exploiter_max_interval, "EXPLOITER_MAX_INTERVAL"),
                      (args.exploiter_max_episodes, "EXPLOITER_MAX_EPISODES")):
        if arg is not None:
            setattr(C, name, arg)

    out_dir = Path(args.out)
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent(C.STATE_DIM, C.NUM_ACTIONS, device=args.device)
    coll = VecCollector(args.workers, args.envs_per_worker, args.map,
                        replay_dir=replay_dir, replay_every=args.replay_every)
    trainer = Trainer(coll, agent)
    print(f"bonk2/ppo: {args.workers} workers x {args.envs_per_worker} envs "
          f"= {coll.E}, hidden {C.HIDDEN}, K={C.ACTION_REPEAT}, "
          f"gamma {C.GAMMA}, rollout {C.ROLLOUT_STEPS}, device {args.device}")

    if args.load:
        with open(args.load) as f:
            trainer.load_state(json.load(f))
    elif args.warm_actor:
        with open(args.warm_actor) as f:
            obj = json.load(f)
        # Load ONLY the actor (the trained policy). The critic is left fresh --
        # required when the critic architecture changed (e.g. 3-atom -> 4-atom
        # categorical), and correct in general for a reward/map change, since a
        # value function calibrated to the old objective would mislead. League
        # and step clocks also stay fresh.
        trainer.agent.actor.load_records(obj["agent"]["actor"])
        print(f"  warm-started actor from {args.warm_actor} "
              f"(fresh league, progress reset)")
        print(f"resumed: episode {trainer.episode_count}, "
              f"ELO {trainer.league.current_rating:.1f}, "
              f"snapshots {len(trainer.league.snapshots)}, "
              f"exploiters {len(trainer.league.exploiters)}")

    def save_checkpoint(tag=""):
        path = out_dir / (f"bonk2-ppo-step{trainer.env_steps}"
                          f"-ep{trainer.episode_count}"
                          f"-elo{round(trainer.league.current_rating)}{tag}.json")
        # Write to a temp file then rename, so a kill mid-write cannot leave a
        # truncated/0-byte checkpoint (which broke --load twice).
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(trainer.serialize()))
        tmp.replace(path)
        print(f"checkpoint saved: {path}")
        # Auto-prune: these checkpoints embed the whole league (~780 MB each),
        # so keep only the most recent CHECKPOINT_KEEP and the -final one.
        if not tag:
            ckpts = sorted(out_dir.glob("bonk2-ppo-step*.json"),
                           key=lambda q: q.stat().st_mtime)
            regular = [q for q in ckpts if "-final" not in q.name]
            for old in regular[:-C.CHECKPOINT_KEEP]:
                old.unlink(missing_ok=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    t_start = time.perf_counter()
    last_log, last_steps, last_saved = t_start, 0, trainer.env_steps
    last_perf = dict(trainer.perf)
    last_stats = None

    try:
        # --steps is the primary stopping unit; --episodes remains as a cap.
        def _running():
            if args.steps:
                return trainer.env_steps < args.steps
            return trainer.episode_count < args.episodes
        while _running() and not stop["flag"]:
            trainer.cycle()
            if trainer.learner_steps() >= C.ROLLOUT_STEPS:
                last_stats, _ = trainer.run_update()

            now = time.perf_counter()
            if now - last_log >= 5.0:
                lg = trainer.league
                p = trainer.perf
                dc = max(1, p["cycles"] - last_perf["cycles"])
                sps = (trainer.env_steps - last_steps) / (now - last_log)
                loss = (f"a={last_stats['actor_loss']:.4f} "
                        f"c={last_stats['critic_loss']:.4f} "
                        f"H={last_stats['entropy']:.3f} "
                        f"ec={last_stats['ent_coef']:.3f}") if last_stats else "warmup"
                phase = ("MAIN" if lg.phase == "main"
                         else f"EXPL {lg.phase_steps/1e6:.1f}M/{C.EXPLOITER_MAX_STEPS/1e6:.0f}M"
                              f"{'*' if lg.exp_gate_hit_at is not None else ''}"
                              f" wr {lg.exp_win_rate()*100:.0f}%")
                top_wr, top_id, losing = lg.pool_status()
                gap = lg.main_ep_total - lg.last_exploiter_ep
                exp_top = lg.top_exploiter_winrate()
                expstr = f"expTop {exp_top*100:.0f}%" if exp_top is not None else "expTop --"
                topstr = (f"topWR {top_wr*100:.0f}% ({top_id}) {expstr} lose {losing} "
                          f"gap {gap//1000}k" if top_wr is not None
                          else f"topWR -- {expstr} gap {gap//1000}k")
                print(f"step {trainer.env_steps:,} (ep {trainer.episode_count:,}) [{phase}|"
                      f"snap {len(lg.snapshots)} exp {len(lg.exploiters)}] | "
                      f"steps/s {sps:.0f} | "
                      f"ELO {lg.current_rating:.1f} | "
                      f"wr {trainer.win_rate()*100:.0f}% | {topstr} | "
                      f"updates {trainer._active_agent().updates} | {loss} | "
                      f"perf/cycle wait={(p['wait']-last_perf['wait'])/dc*1000:.2f} "
                      f"infer={(p['infer']-last_perf['infer'])/dc*1000:.2f}ms "
                      f"train={(p['train']-last_perf['train'])/(now-last_log)*100:.0f}%")
                last_log, last_steps, last_perf = now, trainer.env_steps, dict(p)

            if trainer.env_steps - last_saved >= args.save_every:
                last_saved = trainer.env_steps
                save_checkpoint()
    except Exception:
        save_checkpoint("-crash")   # never lose progress to a bug
        raise
    finally:
        coll.stop()
    save_checkpoint("-final")
    print(f"done: {trainer.episode_count} episodes in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
