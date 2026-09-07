"""bonk2 trainer: multiprocess self-play PPO with league opponents.

Workers collect (collect.VecCollector), the main process infers, learns, and
does the bookkeeping; all population logic lives in league.League. Run:

  python -m bonk4.train --workers 10 --device mps

Device strategy (measured on Apple Silicon at E=2560, [512,512]): the PPO
update is ~2x faster on MPS and big-batch inference is a wash, so --device mps
puts the trained agents (main / exploiter / frozen_main) there; frozen POOL
nets always stay on CPU, where many small per-net batches beat GPU dispatch
overhead. Plain --device cpu remains fully supported.

Ctrl-C saves a checkpoint and shuts the workers down. Checkpoints are the same
JSON shape as v1 (stateDim: 34), so bonk.export_model and the record-based
loaders (play3.mjs, bonk4.play, bonk4.eval_h2h) work unchanged.
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
from .env import (OUT_DRAW, OUT_LOSS, OUT_WIN, mirror_action_batch,
                  mirror_action_into, mirror_obs_batch, mirror_obs_into)
from .league import League
from .pool import EpisodePool, actor_rows, critic_batches
from .ppo import PPOAgent, entropy_coef_at

REPO = Path(__file__).resolve().parents[2]


def make_agent(device: str):
    """ppo4 has exactly one architecture: feedforward actor + prefix-GRU
    outcome critic. The knob this used to read (C.RECURRENT) is gone."""
    return PPOAgent(C.STATE_DIM, C.NUM_ACTIONS, device=device)


class Trainer:
    """PPO over a VecCollector: per-env streams/pendings live here; physics and
    episode resets live in the workers; the population lives in the League."""

    def __init__(self, collector: VecCollector, agent: PPOAgent):
        self.coll = collector
        self.agent = agent              # the MAIN agent (always what gets saved)
        self.league = League(agent)
        self.E = collector.E

        self.episode_count = 0
        self.rng = np.random.default_rng()
        self.env_steps = 0
        self.recent = []                # main's last 100 scores (any opponent)
        self.perf = {"wait": 0.0, "infer": 0.0, "train": 0.0, "cycles": 0}

        # Rollout buffers. Every env advances one row per cycle (pendings are
        # completed in lockstep), so streams are flat [T, E] arrays and GAE
        # vectorizes across all envs — per-env Python lists were the main-
        # process bottleneck at E≈2560.
        E, D = self.E, agent.state_dim
        # RING of the last max_ep rows per env. An episode can never exceed
        # MAX_EPISODE_STEPS, so a ring that long always holds the whole of
        # whatever episode each env is currently in.
        self.max_ep = -(-C.MAX_EPISODE_STEPS // C.ACTION_REPEAT)
        self.RING = self.max_ep + 2
        R_, D_ = self.RING, D
        self.r_s = np.zeros((R_, E, D_), dtype=np.float32)
        self.r_a = np.zeros((R_, E), dtype=np.int64)
        self.r_lp = np.zeros((R_, E), dtype=np.float32)
        self.r_v = np.zeros((R_, E), dtype=np.float32)
        self.r_os = np.zeros((R_, E, D_), dtype=np.float32)
        self.r_oa = np.zeros((R_, E), dtype=np.int64)
        self.r_olp = np.zeros((R_, E), dtype=np.float32)
        self.r_ov = np.zeros((R_, E), dtype=np.float32)
        self.w = 0                                  # ring write row
        self.ep_len = np.zeros(E, dtype=np.int64)   # rows in the live episode
        # Pools of FINISHED episodes waiting for an update.
        cap = C.ROLLOUT_STEPS + self.max_ep * 4 + 16
        self.pool = EpisodePool(cap, D)
        self.opool = EpisodePool(cap, D)
        # Actor scratch, allocated ONCE (variable-size rebuilds every update
        # fragmented malloc and read as an ~18 MB/update leak). Holds learner +
        # opponent pool rows, then the mirrored copy.
        self.a_cap = int(2.8 * C.ROLLOUT_STEPS)
        self.a_s = np.zeros((self.a_cap, D), dtype=np.float32)
        self.a_a = np.zeros(self.a_cap, dtype=np.int64)
        self.a_lp = np.zeros(self.a_cap, dtype=np.float32)
        self.a_adv = np.zeros(self.a_cap, dtype=np.float32)
        buf_mb = (2 * R_ * E * D_ * 4 + 2 * cap * D_ * 4
                  + self.a_cap * (D_ * 4 + 20)) / 1e6
        n_eps = getattr(C, "ROLLOUT_EPISODES", 0)
        print(f"rollout: pool fires at {n_eps:,} FINISHED episodes "
              f"(cap {C.ROLLOUT_STEPS:,} rows) | E={E} | ring {R_} "
              f"(~{buf_mb:.0f} MB) | staleness ~{E/max(n_eps,1):.2f} updates")
        print(f"  episode-pooled: every row has a true outcome (lab=100% by "
              f"construction), each episode unrolled once from h0=0, "
              f"{C.CRITIC_LENGTH_BUCKETS} length buckets to bound padding.")

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
        # Predicted outcome distribution for the IN-FLIGHT decision, i.e. one
        # step past the end of the window. This is the bootstrap target for
        # rows whose episode has not resolved (config.BOOTSTRAP_TAIL), exactly
        # as GAE would bootstrap from the pending value.

        # Terminal outcome class per row (env.OUT_*), written at the row
        # where the episode ends and propagated backwards at update time.
        # What each seat's OPPONENT actually did — the aux target for the
        # opponent-action head (the other seat's action at the same row).

        # ── critic prefix state ────────────────────────────────────────────
        # The critic is recurrent over the PREFIX, so each seat carries its own
        # live hidden state, zeroed when its episode ends. b_h0/b_oh0 snapshot
        # them at the start of each rollout window so update_critic can replay
        # the exact same sequence; detached, which is what makes this truncated
        # BPTT rather than an ever-growing graph.
        H = agent.critic.state_size   # layers * hidden, carried flat
        self.H = H
        self.h_l = torch.zeros(E, H, device=agent.device)
        self.h_o = torch.zeros(E, H, device=agent.device)
        # State that PRODUCED the in-flight pending. The pending is committed on
        # the NEXT cycle, by which point h_l has already moved on, so the
        # window's true h0 has to be carried alongside it.
        # Kept on-device. The pending's h0 must be captured EVERY cycle (an
        # update can reset the window between this cycle's inference and the
        # commit, so we cannot know in advance which pending lands on row 0),
        # but a device->host copy every cycle is a full MPS pipeline sync for
        # data read once per rollout. Clone on-device, sync only at row 0.

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
        """Pick env i's opponent."""
        if self.league.phase == "exploiter":
            self.opp[i] = ("frozen", None)   # best-respond to the frozen main
            self.cur_mask[i] = False
            return
        if not force and C.OPPONENT_HOLD_EPISODES > 1 and self.opp_left[i] > 0:
            self.opp_left[i] -= 1
            return                            # keep facing the same opponent
        kind, idx = self.league.sample_opponent()
        self.opp[i] = (kind, idx)
        # Mirror match: both seats are the live learner -> both train.
        self.cur_mask[i] = kind == "current"
        self.opp_left[i] = max(0, C.OPPONENT_HOLD_EPISODES - 1)

    def _shift_opponent_indices(self, kind):
        """The oldest (kind) pool member was evicted; in-flight opponents keep
        the same net one index lower."""
        for j in range(self.E):
            if self.opp[j][0] == kind:
                self.opp[j] = (kind, max(0, self.opp[j][1] - 1))

    def spawn_frac(self) -> float:
        """0 -> always start the discs close, 1 -> always the map's own spawns.
        Linear in episodes: the agent needs the close regime for as long as it
        takes to learn contact at all, and a slow fade keeps the two task
        distributions overlapping instead of stepping between them."""
        if not C.SPAWN_CURRICULUM:
            return 1.0
        n = max(1, C.SPAWN_CURRICULUM_EPISODES)
        ceil = min(1.0, self.episode_count / n)
        # SAMPLED per cycle, not annealed. A global anneal means that at any
        # moment the agent trains on exactly ONE spawn difficulty and has zero
        # coverage of the others -- at 1.4M episodes it had never once seen the
        # map's real spawns, which is the only thing a human ever plays it from.
        # Drawing a fraction instead keeps real spawns permanently in the mix
        # while the ceiling still ramps the hard cases in gradually.
        if self.rng.random() < C.SPAWN_REAL_PROB:
            return 1.0
        return float(self.rng.random()) * ceil

    def ready_for_update(self) -> bool:
        """Episode count is the primary trigger (it auto-scales with episode
        length); the row cap is only a memory backstop."""
        n_eps = getattr(C, "ROLLOUT_EPISODES", 0)
        if n_eps and len(self.pool.eps) >= n_eps:
            return True
        return self.pool.n >= C.ROLLOUT_STEPS

    def learner_steps(self):
        """Rows of FINISHED episodes waiting in the pool."""
        return self.pool.n

    # ── one decision-block cycle ───────────────────────────────────────────────
    def cycle(self):
        t0 = time.perf_counter()
        # Workers read this when they reset an env; push it before they block.
        self.coll.spawn_frac.value = self.spawn_frac()
        obs, brew, done = self.coll.sync_obs()
        t1 = time.perf_counter()
        self.env_steps += self.E * self.coll.k

        # 1. Fold the finished block into pendings and commit into the ring.
        ended = done[:, 0] > 0
        if self.p_valid:
            w = self.w
            self.r_s[w] = self.p_s
            self.r_a[w] = self.p_a
            self.r_lp[w] = self.p_lp
            self.r_v[w] = self.p_v
            self.r_os[w] = self.po_s
            self.r_oa[w] = self.po_a
            self.r_olp[w] = self.po_lp
            self.r_ov[w] = self.po_v
            self.ep_len += 1
            self.p_valid = False
            if ended.any():
                idxs = np.nonzero(ended)[0]
                self._harvest(idxs, done, brew)
                for i in idxs:
                    # league bookkeeping (ELO, PFSP, snapshots, exploiter
                    # phases) still runs per finished episode.
                    self._finish_episode(int(i), done[i])
            self.w = (w + 1) % self.RING

        # An ended episode must not leak prefix memory into the next one.
        # This runs AFTER the row was committed (that row still belongs to the
        # finished episode) and BEFORE the next inference.
        if ended.any():
            em = torch.as_tensor(ended, device=self.h_l.device)
            self.h_l[em] = 0.0
            self.h_o[em] = 0.0

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
        # The critic's hidden state rides the SAME merged batch: rows 0..E are
        # the learner seat's prefixes, the rest are the current-opponent seats'.
        # One GRU step advances both streams together.
        cur_t = (torch.as_tensor(cur, device=self.h_l.device) if len(cur)
                 else None)
        h_in = (torch.cat([self.h_l, self.h_o[cur_t]]) if len(cur)
                else self.h_l)
        pad = -n_real % 512
        if pad:
            both = np.concatenate(
                [both, np.zeros((pad, both.shape[1]), dtype=np.float32)])
            h_in = torch.cat([h_in, torch.zeros(pad, self.H, device=h_in.device)])
        # No h0 snapshot needed any more: episodes are replayed from their own
        # start (h0 = 0) at update time, so nothing about the live hidden state
        # has to be persisted.
        acts, lps, vals, prbs, h_out = active.act_batch(both, h_in)
        acts, lps, vals, prbs = (acts[:n_real], lps[:n_real],
                                 vals[:n_real], prbs[:n_real])
        self.h_l = h_out[:self.E].contiguous()
        if len(cur):
            self.h_o[cur_t] = h_out[self.E:n_real].contiguous()
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

    def _harvest(self, idxs, done, brew):
        """Move each just-finished episode out of the ring and into the pool.

        Rows live at ring positions (w - len + 1 .. w), wrapping. The learner
        seat always contributes; the opponent seat only when it was driven by
        the CURRENT policy (otherwise its rows are off-policy for this actor and
        po_* was never even written for it).
        """
        w, R_ = self.w, self.RING
        d0, d1 = done[:, 1] > 0, done[:, 2] > 0
        to_ = done[:, 3] > 0
        for i in idxs:
            k = int(self.ep_len[i])
            self.ep_len[i] = 0
            if k <= 0:
                continue
            sel = (np.arange(w - k + 1, w + 1) % R_)
            draw = to_[i] or (d0[i] and d1[i])
            if draw:
                cls, ocls = OUT_DRAW, OUT_DRAW
            elif d1[i]:
                cls, ocls = OUT_WIN, OUT_LOSS
            else:
                cls, ocls = OUT_LOSS, OUT_WIN
            # brew holds this block's reward, which for a terminal block IS the
            # episode return (every non-terminal reward is 0). That is what
            # carries WIN_TIME_DECAY.
            ret, oret = float(brew[i, 0]), float(brew[i, 1])
            self.pool.add(self.r_s[sel, i], self.r_a[sel, i], self.r_lp[sel, i],
                          self.r_v[sel, i], self.r_oa[sel, i], self.r_a[sel, i],
                          cls, ret)
            if self.po_m[i]:
                self.opool.add(self.r_os[sel, i], self.r_oa[sel, i],
                               self.r_olp[sel, i], self.r_ov[sel, i],
                               self.r_a[sel, i], self.r_oa[sel, i], ocls, oret)

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
        self.pool.clear()
        self.opool.clear()
        self.ep_len[:] = 0
        self.p_valid = False
        self.po_m[:] = False
        self.h_l.zero_()
        self.h_o.zero_()
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

    def run_update(self):
        """Train on the pooled FINISHED episodes.

        No window, no carry, no bootstrap: every pooled row carries its
        episode's true outcome, so the critic gets a one-hot target and the
        actor gets the real Monte-Carlo return. Each episode is one sequence
        starting at h0 = 0.
        """
        t0 = time.perf_counter()
        if self.pool.n < 2:
            return self.agent.stats, 0
        D = self.agent.state_dim
        agent = self._active_agent()

        # ── critic: whole episodes, length-bucketed so padding stays ~1.1x ──
        _t = time.perf_counter()
        batches = list(critic_batches(self.pool, C.CRITIC_LENGTH_BUCKETS, D))
        if self.opool.n:
            batches += list(critic_batches(self.opool, C.CRITIC_LENGTH_BUCKETS, D))
        prep = time.perf_counter() - _t
        _t = time.perf_counter()
        stats = agent.update_critic_episodes(batches)
        vh, hstats = agent.update_hindsight(batches)
        stats.update(hstats)
        tcrit = time.perf_counter() - _t

        # ── actor: flat rows, learner + opponent seat + mirror ──
        _t = time.perf_counter()
        s_l, a_l, lp_l, adv_l = actor_rows(self.pool, vh)
        n = len(a_l)
        self.a_s[:n] = s_l; self.a_a[:n] = a_l
        self.a_lp[:n] = lp_l; self.a_adv[:n] = adv_l
        if self.opool.n:
            s_o, a_o, lp_o, adv_o = actor_rows(self.opool)
            k = min(len(a_o), self.a_cap - n)
            self.a_s[n:n + k] = s_o[:k]; self.a_a[n:n + k] = a_o[:k]
            self.a_lp[n:n + k] = lp_o[:k]; self.a_adv[n:n + k] = adv_o[:k]
            n += k
        if C.MIRROR_MODE == "duplicate" and 2 * n <= self.a_cap:
            mirror_obs_into(self.a_s[n:2 * n], self.a_s[:n])
            mirror_action_into(self.a_a[n:2 * n], self.a_a[:n])
            self.a_lp[n:2 * n] = self.a_lp[:n]
            self.a_adv[n:2 * n] = self.a_adv[:n]
            n *= 2
        ec = (self.league.exploiter_entropy_coef() if self.league.phase == "exploiter"
              else entropy_coef_at(self.league.main_ep_total))
        stats.update(agent.update_actor(
            {"states": self.a_s[:n], "actions": self.a_a[:n],
             "logps": self.a_lp[:n], "advantages": self.a_adv[:n]}, ec))
        agent.updates += 1
        agent.stats = stats

        eps = len(self.pool.eps)
        stats["T"] = int(np.mean([e[1] for e in self.pool.eps])) if eps else 0
        # Explained variance of the baseline: 1 - Var(R-V)/Var(R). THE accuracy
        # number for the critic — unlike outcome_acc it cannot be faked by
        # scoring only the easy near-terminal rows. Lost in the pool refactor;
        # restored here.
        _r = np.empty(self.pool.n, dtype=np.float32)
        for off, k, _c, r_ in self.pool.eps:
            _r[off:off + k] = r_
        _a = _r - self.pool.v[:self.pool.n]
        _vr = float(_r.var())
        stats["ev"] = float(1.0 - _a.var() / _vr) if _vr > 1e-8 else 0.0
        stats["labelled"] = 1.0            # pooled rows are all real outcomes
        stats["exact"] = 1.0
        stats["tprep"], stats["tcrit"] = prep, tcrit
        stats["tact"] = time.perf_counter() - _t
        rows = self.pool.n
        self.pool.clear()
        self.opool.clear()
        self.perf["train"] += time.perf_counter() - t0
        return stats, rows

    def win_rate(self):
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    # ── save / load ────────────────────────────────────────────────────────────
    def serialize(self):
        return {
            "version": 2,
            "algo": "ppo",
            "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stateDim": self.agent.state_dim,
            "arch": "ppo4-prefix-critic",
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
        got = obj.get("arch", "feedforward")
        if got != "ppo4-prefix-critic":
            raise ValueError(
                f"architecture mismatch: checkpoint is '{got}', ppo4 needs "
                f"'ppo4-prefix-critic'. ppo2/ppo3 checkpoints have a regression "
                f"critic and different reward scaling — start a fresh run.")
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
    print(f"bonk4/ppo: {args.workers} workers x {args.envs_per_worker} envs "
          f"= {coll.E}, actor {C.HIDDEN}, critic enc {C.CRITIC_ENC} "
          f"gru {C.CRITIC_HIDDEN}, K={C.ACTION_REPEAT}, "
          f"rollout {C.ROLLOUT_STEPS}, device {args.device}")
    print(f"critic: prefix-GRU outcome classifier over {C.OUTCOMES} "
          f"= {C.OUTCOME_REWARD}; advantage = R - V(prefix), no GAE/gamma")

    if args.load:
        with open(args.load) as f:
            trainer.load_state(json.load(f))
        print(f"resumed: episode {trainer.episode_count}, "
              f"ELO {trainer.league.current_rating:.1f}, "
              f"snapshots {len(trainer.league.snapshots)}, "
              f"exploiters {len(trainer.league.exploiters)}")

    def save_checkpoint(tag=""):
        path = out_dir / (f"bonk4-ppo-ep{trainer.episode_count}"
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
            if trainer.ready_for_update():
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
                        + (f" self={last_stats['aux_self']:.3f}"
                           if last_stats.get('aux_self') else "")
                        + (f" oacc={last_stats['outcome_acc']*100:.0f}%"
                           if last_stats.get('outcome_acc') else "")
                        + (f" ev={last_stats['ev']:.3f}"
                           if last_stats.get('ev') is not None else "")
                        + (f" lab={last_stats['labelled']*100:.0f}%"
                           + ("!!LOW" if last_stats['labelled'] < 0.5 else "")
                           if last_stats.get('labelled') is not None else "")
                        + (f" exact={last_stats['exact']*100:.0f}%"
                           if last_stats.get('exact') is not None else "")
                        + (f" T={last_stats['T']}" if last_stats.get('T') else "")
                        + (f" hca={last_stats['hca_acc']*100:.0f}%"
                           f"/hadv={last_stats['hadv']:.2f}"
                           if last_stats.get('hca_acc') else "")
                        + (f" [prep {last_stats['tprep']:.1f}s crit "
                           f"{last_stats['tcrit']:.1f}s act {last_stats['tact']:.1f}s]"
                           if last_stats.get('tcrit') else "")
                        + (f" spawn={trainer.spawn_frac()*100:.0f}%"
                           if C.SPAWN_CURRICULUM else "")
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
