"""bonk2 trainer: multiprocess self-play PPO with league opponents.

Workers collect (collect.VecCollector), the main process infers, learns, and
does the bookkeeping; all population logic lives in league.League. Run:

  python -m bonk2.train --workers 10 --device mps

Device strategy (measured on Apple Silicon at E=2560, [512,512]): the PPO
update is ~2x faster on MPS and big-batch inference is a wash, so --device mps
puts the trained agents (main / exploiter / frozen_main) there; frozen POOL
nets always stay on CPU, where many small per-net batches beat GPU dispatch
overhead. Plain --device cpu remains fully supported.

Ctrl-C saves a checkpoint and shuts the workers down. Checkpoints are the same
JSON shape as v1 (stateDim: 34), so bonk.export_model and the record-based
loaders (play3.mjs, bonk2.play, bonk2.eval_h2h) work unchanged.
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
from .recurrent import RecurrentAgent

REPO = Path(__file__).resolve().parents[2]


def make_agent(device: str):
    """One place decides the architecture, so the trainer, the league's
    exploiters and any resume all agree."""
    cls = RecurrentAgent if C.RECURRENT else PPOAgent
    return cls(C.STATE_DIM, C.NUM_ACTIONS, device=device)


class Trainer:
    """PPO over a VecCollector: per-env streams/pendings live here; physics and
    episode resets live in the workers; the population lives in the League."""

    def __init__(self, collector: VecCollector, agent: PPOAgent):
        self.coll = collector
        self.agent = agent              # the MAIN agent (always what gets saved)
        self.league = League(agent)
        self.E = collector.E

        self.episode_count = 0
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

        # ── recurrent state ────────────────────────────────────────────────
        # h_l/h_o are the LIVE hidden states (learner seat / opponent seat),
        # carried across cycles and zeroed when an episode ends. b_h0 snapshots
        # them at the start of each rollout window so the update can replay the
        # exact same sequence; it is detached, which is what makes this
        # truncated BPTT rather than an ever-growing graph.
        self.recurrent = C.RECURRENT
        if self.recurrent:
            H = agent.net.hidden_size
            self.h_l = np.zeros((E, H), dtype=np.float32)
            self.h_o = np.zeros((E, H), dtype=np.float32)
            self.b_h0 = np.zeros((E, H), dtype=np.float32)
            # Hidden state that PRODUCED the in-flight pending. The pending is
            # committed on the NEXT cycle, by which point h_l has already moved
            # on, so the window's true h0 has to be carried alongside it.
            self.p_h = np.zeros((E, H), dtype=np.float32)
            # aux target: what the OPPONENT actually did (their applied action)
            self.b_oppact = np.zeros((self.T, E), dtype=np.int64)
            self.p_oppact = np.zeros(E, dtype=np.int64)

        self.opp = [None] * self.E      # ("current",None)|("snap",i)|("exp",i)|("frozen",None)
        self.cur_mask = np.zeros(E, dtype=bool)    # opp[i] is "current"
        # Opponent persistence: memory is only worth learning if the same
        # opponent sticks around long enough to be identified.
        self.opp_left = np.zeros(E, dtype=np.int32)
        for i in range(self.E):
            self._select_opponent(i, force=True)

    def _active_agent(self):
        return self.league.exp_agent if self.league.phase == "exploiter" else self.agent

    def _select_opponent(self, i, force: bool = False):
        """Pick env i's opponent. Under recurrence the SAME opponent is held for
        OPPONENT_HOLD_EPISODES episodes: adaptation is only learnable if there
        is something stable to adapt to across episodes."""
        if self.league.phase == "exploiter":
            self.opp[i] = ("frozen", None)   # best-respond to the frozen main
            self.cur_mask[i] = False
            return
        if not force and self.recurrent and self.opp_left[i] > 0:
            self.opp_left[i] -= 1
            return                            # keep facing the same opponent
        kind, idx = self.league.sample_opponent()
        self.opp[i] = (kind, idx)
        # Mirror match: both seats are the live learner -> both train.
        self.cur_mask[i] = kind == "current"
        self.opp_left[i] = max(0, C.OPPONENT_HOLD_EPISODES - 1) if self.recurrent else 0

    def _shift_opponent_indices(self, kind):
        """The oldest (kind) pool member was evicted; in-flight opponents keep
        the same net one index lower."""
        for j in range(self.E):
            if self.opp[j][0] == kind:
                self.opp[j] = (kind, max(0, self.opp[j][1] - 1))

    def learner_steps(self):
        return self.t * self.E

    # ── one decision-block cycle ───────────────────────────────────────────────
    def cycle(self):
        t0 = time.perf_counter()
        obs, brew, done = self.coll.sync_obs()
        t1 = time.perf_counter()
        self.env_steps += self.E * self.coll.k

        # 1. Fold the finished block into pendings and commit them as row t.
        ended = done[:, 0] > 0
        if self.p_valid:
            t = self.t
            if t >= self.T:
                # The caller must run_update() as soon as learner_steps()
                # reaches ROLLOUT_STEPS; the buffers only carry a small margin
                # past that. Fail with the reason rather than an IndexError.
                raise RuntimeError(
                    f"rollout buffer overflow (t={t}, capacity={self.T}): "
                    "run_update() was not called when learner_steps() reached "
                    f"ROLLOUT_STEPS={C.ROLLOUT_STEPS}")
            self.p_r += brew[:, 0]
            self.b_s[t] = self.p_s
            self.b_a[t] = self.p_a
            self.b_lp[t] = self.p_lp
            self.b_v[t] = self.p_v
            self.b_r[t] = self.p_r
            self.b_d[t] = ended
            self.po_r += brew[:, 1]
            self.b_os[t] = self.po_s
            self.b_oa[t] = self.po_a
            self.b_olp[t] = self.po_lp
            self.b_ov[t] = self.po_v
            self.b_or[t] = self.po_r
            self.b_om[t] = self.po_m
            if self.recurrent:
                self.b_oppact[t] = self.p_oppact
                if t == 0:
                    # This row opens the BPTT window; its h0 is the state that
                    # produced it, captured before h_l advanced.
                    self.b_h0[:] = self.p_h
            self.t = t + 1
            self.p_valid = False
        for i in np.nonzero(ended)[0]:
            self._finish_episode(int(i), done[i])
            # A phase transition inside this loop resets the buffers; the
            # remaining ended envs still get their league bookkeeping.

        if self.recurrent:
            # An ended episode must not leak memory into the next one. This
            # runs AFTER the row was committed (that row still belongs to the
            # finished episode) and BEFORE the next inference.
            if ended.any():
                self.h_l[ended] = 0.0
                self.h_o[ended] = 0.0
            self._cycle_recurrent(obs)
            self.coll.send_actions(self._actions)
            t2 = time.perf_counter()
            self.perf["wait"] += t1 - t0
            self.perf["infer"] += t2 - t1
            self.perf["cycles"] += 1
            return

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
            net = self.league.opponent_net(kind, idx)
            acts = self.agent.act_actions(net, np.ascontiguousarray(obs[idxs, 1]))
            actions[idxs, 1] = acts

        self.coll.send_actions(actions)
        t2 = time.perf_counter()
        self.perf["wait"] += t1 - t0
        self.perf["infer"] += t2 - t1
        self.perf["cycles"] += 1

    def _cycle_recurrent(self, obs):
        """Recurrent inference for one decision block.

        Both seats keep their own hidden state. The learner's advances with the
        ACTIVE agent; the opponent seat advances with whatever net drives it
        (the live agent for mirror matches, a frozen pool net otherwise) —
        hidden size is shared, so one [E, H] array serves either.
        """
        active = self._active_agent()
        E = self.E
        actions = np.zeros((E, 2), dtype=np.int64)
        self.p_h[:] = self.h_l      # state this decision is made from
        l_states = np.ascontiguousarray(obs[:, 0])
        cur = np.nonzero(self.cur_mask)[0]
        self.po_m[:] = self.cur_mask
        self.po_r[:] = 0.0

        # ONE batch for both seats (same net), padded to a multiple of 512.
        # Two reasons, both learned the hard way on the feedforward path:
        # halves the per-cycle inference calls, and — critically — torch-MPS
        # caches a compiled graph per tensor SHAPE and never evicts it, so a
        # batch whose height is len(cur) (which changes every cycle) leaks
        # memory without bound.
        n_l = E
        both = (np.concatenate([l_states, obs[cur, 1]]) if len(cur) else l_states)
        both_h = (np.concatenate([self.h_l, self.h_o[cur]]) if len(cur) else self.h_l)
        n_real = both.shape[0]
        pad = -n_real % 512
        if pad:
            both = np.concatenate([both, np.zeros((pad, both.shape[1]), np.float32)])
            both_h = np.concatenate([both_h, np.zeros((pad, both_h.shape[1]), np.float32)])
        a, lp, v, h_new = active.act_batch(both, both_h)

        actions[:, 0] = a[:n_l]
        self.p_s[:] = l_states
        self.p_a[:] = a[:n_l]
        self.p_lp[:] = lp[:n_l]
        self.p_v[:] = v[:n_l]
        self.p_r[:] = 0.0
        self.h_l = np.ascontiguousarray(h_new[:n_l])
        if len(cur):
            sl = slice(n_l, n_real)
            actions[cur, 1] = a[sl]
            self.po_s[cur] = both[sl]
            self.po_a[cur] = a[sl]
            self.po_lp[cur] = lp[sl]
            self.po_v[cur] = v[sl]
            self.h_o[cur] = h_new[sl]

        groups = {}
        for i in range(E):
            kind, idx = self.opp[i]
            if kind != "current":
                groups.setdefault((kind, idx), []).append(i)
        for (kind, idx), idxs in groups.items():
            net = self.league.opponent_net(kind, idx)
            ai, hi = self.agent.act_actions(
                net, np.ascontiguousarray(obs[idxs, 1]), self.h_o[idxs])
            actions[idxs, 1] = ai
            self.h_o[idxs] = hi

        # Aux target: the action the opponent actually took this decision.
        self.p_oppact[:] = actions[:, 1]
        self.p_valid = True     # pendings are live; next cycle commits them
        self._actions = actions

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
        if self.recurrent:
            self.h_l[:] = 0.0
            self.h_o[:] = 0.0
        for i in range(self.E):
            self._select_opponent(i, force=True)

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
        """Multi-horizon position change for each seat's OWN disc.

        Returns (targets [T*E, H, 2], mask [T*E, H]) for the H horizons in
        C.AUX_POS_HORIZONS. obs[0:2] is the self block's (x, y) scaled by
        POS_SCALE, so the h-step target is the row-(t+h) self-position minus
        this row's, un-scaled back to metres and then divided by h — mean
        metres per decision, which keeps every horizon on one scale (see the
        config note). For h=1 that division is a no-op, so the first horizon
        matches the old single-step target exactly.

        A row is valid for horizon h only if the whole span t..t+h stays inside
        ONE episode: rows t..t+h-1 must all be non-terminal, and t+h must exist
        in the buffer. Crossing a boundary would make the target a teleport
        between two unrelated episodes. `run` counts how many consecutive
        non-terminal rows start at t, computed by one reverse scan so the check
        stays O(T) rather than O(T*H) window maxima.
        """
        H = len(C.AUX_POS_HORIZONS)
        tgt = np.zeros((T, E, H, 2), dtype=np.float32)
        msk = np.zeros((T, E, H), dtype=bool)
        if T < 2:
            return tgt.reshape(T * E, H, 2), msk.reshape(T * E, H)
        alive = buf_d[:T] == 0.0
        run = np.zeros((T, E), dtype=np.int32)
        run[T - 1] = alive[T - 1]
        for t in range(T - 2, -1, -1):
            run[t] = np.where(alive[t], run[t + 1] + 1, 0)
        for j, h in enumerate(C.AUX_POS_HORIZONS):
            if h >= T:            # horizon longer than the rollout window
                continue
            d = (buf_s[h:T, :, 0:2] - buf_s[:T - h, :, 0:2]) / C.POS_SCALE
            tgt[:T - h, :, j] = d / float(h)
            msk[:T - h, :, j] = run[:T - h] >= h
        return tgt.reshape(T * E, H, 2), msk.reshape(T * E, H)

    def _run_update_recurrent(self):
        """BPTT update. Buffers are already [T, E, ...] — each env IS a
        sequence — so no re-chunking is needed. Only the LEARNER seat trains:
        the opponent seat would need its own hidden-state history replayed with
        its own net, and mixing the two into one sequence batch is wrong.
        """
        t0 = time.perf_counter()
        T, E = self.t, self.E
        if T < 2:
            self.t = 0
            return self.agent.stats, 0

        d = self.b_d[:T]
        last_v = np.where(d[T - 1] > 0, 0.0, self.p_v).astype(np.float32)
        adv, ret = self._gae_columns(self.b_r[:T], self.b_v[:T], d, last_v)

        # aux 1: own next-decision displacement, metres (masked at boundaries)
        pos_t, pos_m = self._aux_targets(self.b_s, self.b_d, T, E)
        H = len(C.AUX_POS_HORIZONS)
        pos_t = pos_t.reshape(T, E, H, 2)
        pos_m = pos_m.reshape(T, E, H)
        # aux 2: the opponent's NEXT action (shift the recorded action back one)
        opp_t = np.zeros((T, E), dtype=np.int64)
        opp_m = np.zeros((T, E), dtype=bool)
        if T >= 2:
            opp_t[:T - 1] = self.b_oppact[1:T]
            opp_m[:T - 1] = d[:T - 1] == 0.0
        # aux 3: does THIS seat die within DEATH_SOON_DECISIONS? A row is only
        # labelled when the outcome is actually observable inside the window.
        die_t = np.zeros((T, E), dtype=np.float32)
        die_m = np.zeros((T, E), dtype=bool)
        W = C.DEATH_SOON_DECISIONS
        for t in range(T):
            hi = min(T, t + W + 1)
            seg = d[t:hi]
            ends = seg.max(axis=0) > 0
            # reward < 0 at the terminal row means this seat lost that episode
            lost = (self.b_r[t:hi] * (d[t:hi] > 0)).min(axis=0) < 0
            die_t[t] = (ends & lost).astype(np.float32)
            die_m[t] = ends | (hi - t > W)   # resolved, or the window fit
        roll = {
            "states": self.b_s[:T], "actions": self.b_a[:T],
            "logps": self.b_lp[:T], "values": self.b_v[:T],
            "returns": ret, "advantages": adv, "dones": d,
            "h0": self.b_h0,
            "aux_pos": pos_t, "aux_pos_mask": pos_m,
            "aux_opp": opp_t, "aux_opp_mask": opp_m,
            "aux_death": die_t, "aux_death_mask": die_m,
        }
        ec = (self.league.exploiter_entropy_coef() if self.league.phase == "exploiter"
              else entropy_coef_at(self.league.main_ep_total))
        stats = self._active_agent().update(roll, entropy_coef=ec)
        self.t = 0
        self.perf["train"] += time.perf_counter() - t0
        return stats, T * E

    def run_update(self):
        if self.recurrent:
            return self._run_update_recurrent()
        t0 = time.perf_counter()
        T, E, D = self.t, self.E, self.agent.state_dim

        # Learner seat: every [t, i] cell is a real transition. Cut-off tails
        # (d[T-1]=0) bootstrap from the in-flight pending value.
        last_v = np.where(self.b_d[T - 1] > 0, 0.0, self.p_v).astype(np.float32)
        adv, ret = self._gae_columns(self.b_r[:T], self.b_v[:T], self.b_d[:T], last_v)
        states = self.b_s[:T].reshape(T * E, D).copy()
        actions = self.b_a[:T].reshape(-1).copy()
        logps = self.b_lp[:T].reshape(-1).copy()
        values = self.b_v[:T].reshape(-1).copy()
        advs, rets = adv.reshape(-1), ret.reshape(-1)
        aux_t, aux_m = self._aux_targets(self.b_s, self.b_d, T, E)

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
            at_o, am_o = self._aux_targets(self.b_os, self.b_d, T, E)
            aux_t = np.concatenate([aux_t, at_o[sel]])
            aux_m = np.concatenate([aux_m, am_o[sel]])
        if C.MIRROR_MODE == "duplicate":
            states = np.concatenate([states, mirror_obs_batch(states)])
            actions = np.concatenate([actions, mirror_action_batch(actions)])
            logps, values = np.tile(logps, 2), np.tile(values, 2)
            advs, rets = np.tile(advs, 2), np.tile(rets, 2)
            # A mirrored world moves the opposite way in x.
            aux_mir = aux_t.copy()
            aux_mir[:, :, 0] *= -1.0        # every horizon's x flips
            aux_t = np.concatenate([aux_t, aux_mir])
            # np.tile on the now-2D mask would tile along the HORIZON axis.
            aux_m = np.concatenate([aux_m, aux_m])
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
            "advantages": advs,
            "returns": rets,
            "aux_target": aux_t,
            "aux_mask": aux_m,
        }
        if self.league.phase == "exploiter":
            ec = self.league.exploiter_entropy_coef()
        else:
            ec = entropy_coef_at(self.league.main_ep_total)
        stats = self._active_agent().update(rollout, entropy_coef=ec)
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
            "arch": "recurrent" if self.recurrent else "feedforward",
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
        want = "recurrent" if self.recurrent else "feedforward"
        got = obj.get("arch", "feedforward")
        if got != want:
            raise ValueError(f"architecture mismatch: checkpoint is {got}, "
                             f"config.RECURRENT gives {want}. Recurrent and "
                             f"feedforward weights are not interchangeable.")
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
    ap.add_argument("--out", default=str(REPO / "runs/ppo2"))
    ap.add_argument("--episodes", type=int, default=100_000_000)
    ap.add_argument("--save-every", type=int, default=400_000)
    ap.add_argument("--replay-every", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs-per-worker", type=int, default=320)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="cap torch CPU threads (0 = leave default)")
    ap.add_argument("--exploiter-trigger-wr", type=float,
                    help="override config.EXPLOITER_TRIGGER_WR")
    ap.add_argument("--exploiter-max-interval", type=int,
                    help="override config.EXPLOITER_MAX_INTERVAL")
    ap.add_argument("--exploiter-max-episodes", type=int,
                    help="override config.EXPLOITER_MAX_EPISODES")
    args = ap.parse_args()

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

    agent = make_agent(args.device)
    coll = VecCollector(args.workers, args.envs_per_worker, args.map,
                        replay_dir=replay_dir, replay_every=args.replay_every)
    trainer = Trainer(coll, agent)
    print(f"bonk2/ppo: {args.workers} workers x {args.envs_per_worker} envs "
          f"= {coll.E}, hidden {C.HIDDEN}, K={C.ACTION_REPEAT}, "
          f"gamma {C.GAMMA}, rollout {C.ROLLOUT_STEPS}, device {args.device}")

    if args.load:
        with open(args.load) as f:
            trainer.load_state(json.load(f))
        print(f"resumed: episode {trainer.episode_count}, "
              f"ELO {trainer.league.current_rating:.1f}, "
              f"snapshots {len(trainer.league.snapshots)}, "
              f"exploiters {len(trainer.league.exploiters)}")

    def save_checkpoint(tag=""):
        path = out_dir / (f"bonk2-ppo-ep{trainer.episode_count}"
                          f"-elo{round(trainer.league.current_rating)}{tag}.json")
        path.write_text(json.dumps(trainer.serialize()))
        print(f"checkpoint saved: {path}")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    t_start = time.perf_counter()
    # Seed last_steps from the CURRENT counter, not 0: after --load, env_steps
    # is restored from the checkpoint (billions), so a 0 baseline makes the
    # first steps/s reading absurd.
    last_log, last_steps, last_saved = t_start, trainer.env_steps, trainer.episode_count
    last_perf = dict(trainer.perf)
    last_stats = None

    try:
        while trainer.episode_count < args.episodes and not stop["flag"]:
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
                        f"ec={last_stats['ent_coef']:.3f}"
                        + (f" aux={last_stats['aux_loss']:.4f}"
                           if last_stats.get('aux_loss') else "")
                        + (f" opp={last_stats['aux_opp']:.3f}"
                           if last_stats.get('aux_opp') else "")
                        + (f" die={last_stats['aux_death']:.3f}"
                           if last_stats.get('aux_death') else "")
                        + (f" kl={last_stats['kl']:.4f}"
                           if last_stats.get('kl') is not None else "")
                        + (f" b={last_stats['kl_coef']:.3g}"
                           if last_stats.get('kl_coef') else "")
                        # only shown when the update was cut short, so a quiet
                        # log means TARGET_KL never bound
                        + (f" KLSTOP@{last_stats['epochs']}/{C.EPOCHS}"
                           if last_stats.get('kl_stopped') else "")
                        ) if last_stats else "warmup"
                phase = ("MAIN" if lg.phase == "main"
                         else f"EXPL {lg.phase_eps}/{C.EXPLOITER_MAX_EPISODES}"
                              f"{'*' if lg.exp_gate_hit_at is not None else ''}"
                              f" wr {lg.exp_win_rate()*100:.0f}%")
                top_wr, top_id, losing = lg.pool_status()
                gap = lg.main_ep_total - lg.last_exploiter_ep
                exp_top = lg.top_exploiter_winrate()
                expstr = f"expTop {exp_top*100:.0f}%" if exp_top is not None else "expTop --"
                topstr = (f"topWR {top_wr*100:.0f}% ({top_id}) {expstr} lose {losing} "
                          f"gap {gap//1000}k" if top_wr is not None
                          else f"topWR -- {expstr} gap {gap//1000}k")
                print(f"ep {trainer.episode_count} [{phase}|"
                      f"snap {len(lg.snapshots)} exp {len(lg.exploiters)}] | "
                      f"steps/s {sps:.0f} | "
                      f"ELO {lg.current_rating:.1f} | "
                      f"wr {trainer.win_rate()*100:.0f}% | {topstr} | "
                      f"updates {trainer._active_agent().updates} | {loss} | "
                      f"perf/cycle wait={(p['wait']-last_perf['wait'])/dc*1000:.2f} "
                      f"infer={(p['infer']-last_perf['infer'])/dc*1000:.2f}ms "
                      f"train={(p['train']-last_perf['train'])/(now-last_log)*100:.0f}%")
                last_log, last_steps, last_perf = now, trainer.env_steps, dict(p)

            if trainer.episode_count - last_saved >= args.save_every:
                last_saved = trainer.episode_count
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
