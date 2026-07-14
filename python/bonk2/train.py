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
from .ppo import PPOAgent, compute_gae, entropy_coef_at

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
        self.env_steps = 0
        self.recent = []                # main's last 100 scores (any opponent)
        self.perf = {"wait": 0.0, "infer": 0.0, "train": 0.0, "cycles": 0}

        # Per-env collection state. pend_* = [s, a, logp, v, reward].
        self.opp = [None] * self.E      # ("current",None)|("snap",i)|("exp",i)|("frozen",None)
        self.pend_l = [None] * self.E
        self.pend_o = [None] * self.E
        self.stream_l = [self._stream() for _ in range(self.E)]
        self.stream_o = [None] * self.E
        for i in range(self.E):
            self._select_opponent(i)

    @staticmethod
    def _stream():
        return {"s": [], "a": [], "lp": [], "v": [], "r": [], "d": []}

    def _active_agent(self):
        return self.league.exp_agent if self.league.phase == "exploiter" else self.agent

    def _select_opponent(self, i):
        if self.league.phase == "exploiter":
            self.opp[i] = ("frozen", None)   # best-respond to the frozen main
            self.stream_o[i] = None
        else:
            kind, idx = self.league.sample_opponent()
            self.opp[i] = (kind, idx)
            # Mirror match: both seats are the live learner -> both train.
            self.stream_o[i] = self._stream() if kind == "current" else None
        self.pend_o[i] = None

    def _shift_opponent_indices(self, kind):
        """The oldest (kind) pool member was evicted; in-flight opponents keep
        the same net one index lower."""
        for j in range(self.E):
            if self.opp[j][0] == kind:
                self.opp[j] = (kind, max(0, self.opp[j][1] - 1))

    @staticmethod
    def _complete(stream, pend, done):
        s, a, lp, v, r = pend
        stream["s"].append(s)
        stream["a"].append(a)
        stream["lp"].append(lp)
        stream["v"].append(v)
        stream["r"].append(r)
        stream["d"].append(1.0 if done else 0.0)

    def learner_steps(self):
        return sum(len(s["s"]) for s in self.stream_l)

    # ── one decision-block cycle ───────────────────────────────────────────────
    def cycle(self):
        t0 = time.perf_counter()
        obs, brew, done = self.coll.sync_obs()
        t1 = time.perf_counter()
        self.env_steps += self.E * self.coll.k

        # 1. Fold the finished block into pendings; close episodes.
        for i in range(self.E):
            ended = done[i, 0] > 0
            if self.pend_l[i] is not None:
                self.pend_l[i][4] += brew[i, 0]
                self._complete(self.stream_l[i], self.pend_l[i], ended)
                self.pend_l[i] = None
            if self.pend_o[i] is not None:
                self.pend_o[i][4] += brew[i, 1]
                self._complete(self.stream_o[i], self.pend_o[i], ended)
                self.pend_o[i] = None
            if ended:
                self._finish_episode(i, done[i])

        # 2. Batched inference. The learner seat uses the ACTIVE agent (main, or
        #    the exploiter during its phase); opponents route by kind.
        active = self._active_agent()
        actions = np.zeros((self.E, 2), dtype=np.int64)
        # Learner seats + "current" opponent seats share the same net in main
        # phase (and cur is empty in exploiter phase), so run them as ONE batch.
        cur = [i for i in range(self.E) if self.opp[i][0] == "current"]
        l_states = np.ascontiguousarray(obs[:, 0])
        both = (np.concatenate([l_states, obs[cur, 1]]) if cur else l_states)
        acts, lps, vals = active.act_batch(both)
        actions[:, 0] = acts[:self.E]
        for i in range(self.E):
            self.pend_l[i] = [l_states[i].copy(), int(acts[i]),
                              float(lps[i]), float(vals[i]), 0.0]
        for j, i in enumerate(cur):
            k = self.E + j
            actions[i, 1] = acts[k]
            self.pend_o[i] = [both[k].copy(), int(acts[k]),
                              float(lps[k]), float(vals[k]), 0.0]
        # Frozen opponents grouped so each net does one batched forward.
        groups = {}
        for i in range(self.E):
            kind, idx = self.opp[i]
            if kind != "current":
                groups.setdefault((kind, idx), []).append(i)
        for (kind, idx), idxs in groups.items():
            net = self.league.opponent_net(kind, idx)
            acts = self.agent.act_actions(net, np.ascontiguousarray(obs[idxs, 1]))
            for j, i in enumerate(idxs):
                actions[i, 1] = acts[j]

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
        """Phase changed: drop all in-flight streams/pendings (they belong to
        the old learner) and re-pick every opponent."""
        for i in range(self.E):
            self.pend_l[i] = self.pend_o[i] = None
            self.stream_l[i] = self._stream()
            self.stream_o[i] = None
            self._select_opponent(i)

    # ── PPO update ─────────────────────────────────────────────────────────────
    def run_update(self):
        t0 = time.perf_counter()
        S, A, LP, V, ADV, RET = [], [], [], [], [], []
        for i in range(self.E):
            for stream, pend in ((self.stream_l[i], self.pend_l[i]),
                                 (self.stream_o[i], self.pend_o[i])):
                if stream is None or not stream["s"]:
                    continue
                # Bootstrap a cut-off tail with the pending value estimate.
                last_v = pend[3] if (pend is not None and not stream["d"][-1]) else 0.0
                adv, ret = compute_gae(stream["r"], stream["v"], stream["d"], last_v)
                S.append(np.stack(stream["s"]))
                A.append(np.asarray(stream["a"], dtype=np.int64))
                LP.append(np.asarray(stream["lp"], dtype=np.float32))
                V.append(np.asarray(stream["v"], dtype=np.float32))
                ADV.append(adv)
                RET.append(ret)
        states = np.concatenate(S).astype(np.float32)
        actions = np.concatenate(A)
        logps, values = np.concatenate(LP), np.concatenate(V)
        advs, rets = np.concatenate(ADV), np.concatenate(RET)
        if C.MIRROR_AUGMENT:
            states = np.concatenate([states, mirror_obs_batch(states)])
            actions = np.concatenate([actions, mirror_action_batch(actions)])
            logps, values = np.tile(logps, 2), np.tile(values, 2)
            advs, rets = np.tile(advs, 2), np.tile(rets, 2)
        rollout = {
            "states": states,
            "actions": actions,
            "logps": logps,
            "values": values,
            "advantages": advs,
            "returns": rets,
        }
        if self.league.phase == "exploiter":
            ec = self.league.exploiter_entropy_coef()
        else:
            ec = entropy_coef_at(self.league.main_ep_total)
        stats = self._active_agent().update(rollout, entropy_coef=ec)
        for i in range(self.E):
            self.stream_l[i] = self._stream()
            if self.stream_o[i] is not None:
                self.stream_o[i] = self._stream()
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
    ap.add_argument("--out", default=str(REPO / "runs/ppo2"))
    ap.add_argument("--episodes", type=int, default=100_000_000)
    ap.add_argument("--save-every", type=int, default=1_000_000)
    ap.add_argument("--replay-every", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--envs-per-worker", type=int, default=256)
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
    last_log, last_steps, last_saved = t_start, 0, trainer.episode_count
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
                        f"ec={last_stats['ent_coef']:.3f}") if last_stats else "warmup"
                phase = ("MAIN" if lg.phase == "main"
                         else f"EXPL {lg.phase_eps}/{C.EXPLOITER_MAX_EPISODES}"
                              f"{'*' if lg.exp_gate_hit_at is not None else ''}"
                              f" wr {lg.exp_win_rate()*100:.0f}%")
                top_wr, top_id, losing = lg.pool_status()
                gap = lg.main_ep_total - lg.last_exploiter_ep
                topstr = (f"topWR {top_wr*100:.0f}% ({top_id}) lose {losing} "
                          f"gap {gap//1000}k" if top_wr is not None
                          else f"topWR -- gap {gap//1000}k")
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
    finally:
        coll.stop()
    save_checkpoint("-final")
    print(f"done: {trainer.episode_count} episodes in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
