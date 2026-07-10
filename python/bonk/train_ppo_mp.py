"""Multiprocess self-play PPO: worker processes collect (mp_collect), the main
process infers, learns, and does the self-play bookkeeping. Same algorithm,
checkpoint format, and hyperparameters as train_ppo — just parallel collection.

  python -m bonk.train_ppo_mp --workers 6 --envs-per-worker 48

Ctrl-C saves a checkpoint and shuts the workers down.
"""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .mp_collect import VecCollector
from .networks import MLP
from .ppo import PPOAgent, HIDDEN, entropy_coef_at
from .train_ppo import (INITIAL_RATING, MIRROR_AUGMENT,
                        OPPONENT_CURRENT_PROB, ROLLOUT_STEPS, SNAPSHOT_BUFFER,
                        SNAPSHOT_INTERVAL, compute_gae)
from .trainer import elo_update, mirror_action, mirror_obs

REPO = Path(__file__).resolve().parents[2]

# --- League (AlphaStar-lite) ---------------------------------------------------
# Every EXPLOITER_INTERVAL main-agent episodes, main training PAUSES and a fresh
# exploiter agent trains against the frozen current main (pure best-response —
# it exists to find the main agent's weaknesses). The finished exploiter's actor
# joins the exploiter pool, and main training resumes with opponents drawn from
# { current model, past snapshot, exploiter }, exploiters weighted by winrate.
# Playing against its own exploiters forces the main agent to patch the exact
# holes a dedicated adversary found — the league antidote to self-play cycling.
EXPLOITER_INTERVAL = 200_000
# Exploiter phase length is adaptive (AlphaStar's recipe: winrate gate +
# timeout, not a fixed budget). Train at least MIN episodes; once the rolling
# winrate vs the frozen main reaches TARGET_WR, sharpen for EXTRA more episodes
# at low entropy and stop. MAX is the give-up timeout — a phase that never
# finds a real exploit shouldn't keep eating main-agent training time.
EXPLOITER_MIN_EPISODES = 20_000
EXPLOITER_MAX_EPISODES = 500_000
EXPLOITER_TARGET_WR = 0.75
EXPLOITER_EXTRA_EPISODES = 10_000
EXPLOITER_WR_WINDOW = 1000    # rolling winrate window (episodes)
EXPLOITER_WARM_START = True   # exploiter starts from main's weights (faster);
                              # False = from scratch (weirder, slower exploits)
EXPLOITER_POOL_MAX = 30
# Exploiter entropy: with WARM_START, EC_START must stay LOW — a high entropy
# bonus melts the transferred policy back to ~uniform (H -> ln(NUM_ACTIONS))
# within a few thousand episodes, throwing away the warm start and forcing a
# de-facto cold restart. Keep just enough entropy to bend the inherited policy
# toward the exploit. (Cold start? Then raise this back to ~0.3.)
EXPLOITER_EC_START = 0.05
EXPLOITER_EC_END = 0.01
EXPLOITER_EC_DECAY_EPISODES = 40_000


class MPPPOTrainer:
    """PPO self-play over a VecCollector: per-env streams/pendings live here;
    physics and episode resets live in the workers."""

    def __init__(self, collector: VecCollector, agent: PPOAgent):
        self.coll = collector
        self.agent = agent          # the MAIN agent (always what gets saved)
        self.E = collector.E

        self.snapshots = []
        self.current_rating = INITIAL_RATING
        self.episode_count = 0
        self.env_steps = 0
        self.recent = []
        self.perf = {"wait": 0.0, "infer": 0.0, "train": 0.0, "cycles": 0}

        # League state.
        self.exploiters = []            # {net, episode, winrate}
        self.phase = "main"             # "main" | "exploiter"
        self.exp_agent = None           # PPOAgent while an exploiter trains
        self.frozen_main = None         # frozen main actor (exploiter's opponent)
        self.main_ep_total = 0          # main-phase episodes (drives the interval)
        self.next_exploiter_at = EXPLOITER_INTERVAL
        self.phase_eps = 0              # episodes inside the current exploiter phase
        self.exp_recent = []            # exploiter scores vs main (rolling)
        self.exp_gate_hit_at = None     # phase_eps when TARGET_WR was reached

        self.opp = [None] * self.E       # ("current", None) | ("snap", idx)
        self.pend_l = [None] * self.E    # [s, a, logp, v, reward]
        self.pend_o = [None] * self.E
        self.stream_l = [self._stream() for _ in range(self.E)]
        self.stream_o = [None] * self.E
        for i in range(self.E):
            self._select_opponent(i)

    @staticmethod
    def _stream():
        return {"s": [], "a": [], "lp": [], "v": [], "r": [], "d": []}

    def _active_agent(self):
        return self.exp_agent if self.phase == "exploiter" else self.agent

    def _select_opponent(self, i):
        if self.phase == "exploiter":
            # Exploiter always best-responds to the frozen main.
            self.opp[i] = ("frozen", None)
            self.stream_o[i] = None
            self.pend_o[i] = None
            return
        # Main phase: pick an opponent KIND, then an instance. Normally the kinds
        # are weighted uniformly, but when a dominant exploiter exists (the main
        # currently loses badly to some pool member) we over-sample the exploiter
        # kind so the main gets enough exposure to actually patch the hole — a
        # 1-in-3 diet is too little against a 90%+ exploiter.
        kinds = ["current"]
        kw = [1.0]
        if self.snapshots:
            kinds.append("snap")
            kw.append(1.0)
        if self.exploiters:
            kinds.append("exp")
            # Weight = how hard the scariest live exploiter beats the main. At
            # ~50% (main holds its own) this is ~1 (uniform); at 90% it's ~3x.
            top = max(self._exp_weight(e) for e in self.exploiters)
            kw.append(1.0 + 4.0 * max(0.0, top - 0.5))
        kw = np.array(kw)
        kind = kinds[np.random.choice(len(kinds), p=kw / kw.sum())]
        if kind == "current":
            self.opp[i] = ("current", None)
            self.stream_o[i] = self._stream()
        elif kind == "snap":
            # PFSP over past selves: concentrate the snapshot budget on the
            # versions the main currently LOSES to, instead of spreading it
            # uniformly (which dilutes each of 50 snapshots to ~0.6% and lets
            # the main cycle freely). Beating your own past is the fictitious-
            # play pressure that turns cycling into monotonic improvement.
            w = np.array([self._snap_weight(s) for s in self.snapshots])
            self.opp[i] = ("snap", int(np.random.choice(len(self.snapshots),
                                                        p=w / w.sum())))
            self.stream_o[i] = None
        else:
            # PFSP-style: sample exploiters by how often the main STILL loses to
            # them (live, not frozen). Once the main patches a hole the exploiter
            # exposed, that exploiter's live winrate collapses, its weight drops,
            # and the main moves on to unsolved ones. Falls back to the
            # graduation winrate until an exploiter has been played enough.
            w = np.array([self._exp_weight(e) for e in self.exploiters])
            self.opp[i] = ("exp", int(np.random.choice(len(self.exploiters),
                                                       p=w / w.sum())))
            self.stream_o[i] = None
        self.pend_o[i] = None

    def _snap_weight(self, s):
        """Live sampling weight for a past-self snapshot = the main's recent
        loss rate against it (floored so solved ones still get sampled). Neutral
        default until enough games exist, so new snapshots gather data first."""
        r = s["recent"]
        if len(r) >= 30:
            return max(1.0 - sum(r) / len(r), 0.1)   # main loss rate vs this self
        return 0.5

    def _exp_weight(self, e):
        """Live exploiter strength = main's recent loss rate against it (floored
        so a solved exploiter never fully vanishes). Graduation winrate until
        the main has enough recent games."""
        if len(e["recent"]) >= 30:
            exp_wr = 1.0 - sum(e["recent"]) / len(e["recent"])   # main loss rate
        else:
            exp_wr = e["winrate"]
        return max(exp_wr, 0.15)

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

    # --- one decision-block cycle ------------------------------------------------
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
        l_states = np.ascontiguousarray(obs[:, 0])
        acts, lps, vals = active.act_batch(l_states)
        actions[:, 0] = acts
        for i in range(self.E):
            self.pend_l[i] = [l_states[i].copy(), int(acts[i]),
                              float(lps[i]), float(vals[i]), 0.0]

        cur = [i for i in range(self.E) if self.opp[i][0] == "current"]
        if cur:
            o_states = np.ascontiguousarray(obs[cur, 1])
            acts, lps, vals = self.agent.act_batch(o_states)
            for j, i in enumerate(cur):
                actions[i, 1] = acts[j]
                self.pend_o[i] = [o_states[j].copy(), int(acts[j]),
                                  float(lps[j]), float(vals[j]), 0.0]
        # Frozen nets: past snapshots, exploiters, and (in exploiter phase) the
        # frozen main — grouped so each net does one batched forward.
        groups = {}
        for i in range(self.E):
            kind, idx = self.opp[i]
            if kind == "snap":
                groups.setdefault(("snap", idx), []).append(i)
            elif kind == "exp":
                groups.setdefault(("exp", idx), []).append(i)
            elif kind == "frozen":
                groups.setdefault(("frozen", 0), []).append(i)
        for (kind, idx), idxs in groups.items():
            net = (self.snapshots[idx]["net"] if kind == "snap"
                   else self.exploiters[idx]["net"] if kind == "exp"
                   else self.frozen_main)
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

        if self.phase == "exploiter":
            self.exp_recent.append(score)
            del self.exp_recent[:-EXPLOITER_WR_WINDOW]
            self.phase_eps += 1
            if self._exploiter_phase_done():
                self._finish_exploiter()
            else:
                self._select_opponent(i)
            return

        # --- main phase --------------------------------------------------------
        self.recent.append(score)
        del self.recent[:-100]
        kind, snap_i = self.opp[i]
        if kind == "snap":
            item = self.snapshots[snap_i]
            self.current_rating, item["rating"] = elo_update(
                self.current_rating, item["rating"], score)
            item["recent"].append(score)   # main's score vs this past self
            del item["recent"][:-200]
        elif kind == "exp":
            rec = self.exploiters[snap_i]["recent"]
            rec.append(score)          # main's score vs this exploiter (1=main won)
            del rec[:-200]

        self.main_ep_total += 1
        if self.main_ep_total % SNAPSHOT_INTERVAL == 0:
            net = MLP(self.agent.state_dim, HIDDEN, self.agent.num_actions)
            net.load_state_dict(self.agent.actor.state_dict())
            net.to(self.agent.device)
            for p in net.parameters():
                p.requires_grad_(False)
            self.snapshots.append({"net": net, "rating": self.current_rating,
                                   "episode": self.episode_count, "recent": []})
            if len(self.snapshots) > SNAPSHOT_BUFFER:
                self.snapshots.pop(0)
                for j in range(self.E):
                    if self.opp[j][0] == "snap":
                        self.opp[j] = ("snap", max(0, self.opp[j][1] - 1))

        if self.main_ep_total >= self.next_exploiter_at:
            self._start_exploiter()
        else:
            self._select_opponent(i)

    # --- league phase transitions ---------------------------------------------
    def _exploiter_phase_done(self):
        """AlphaStar-style adaptive stop: winrate gate + sharpen tail + timeout."""
        if self.phase_eps >= EXPLOITER_MAX_EPISODES:
            return True
        if self.phase_eps < EXPLOITER_MIN_EPISODES:
            return False
        if self.exp_gate_hit_at is None:
            if (len(self.exp_recent) >= EXPLOITER_WR_WINDOW
                    and self.exp_win_rate() >= EXPLOITER_TARGET_WR):
                self.exp_gate_hit_at = self.phase_eps
                print(f"=== league: exploiter hit "
                      f"{EXPLOITER_TARGET_WR * 100:.0f}% at {self.phase_eps} eps "
                      f"— sharpening for {EXPLOITER_EXTRA_EPISODES} more ===")
            return False
        return self.phase_eps >= self.exp_gate_hit_at + EXPLOITER_EXTRA_EPISODES

    def _freeze(self, src_net):
        net = MLP(self.agent.state_dim, HIDDEN, self.agent.num_actions)
        net.load_state_dict(src_net.state_dict())
        net.to(self.agent.device)
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    def _reset_collection(self):
        """Drop all in-flight streams/pendings (they belong to the old learner)."""
        for i in range(self.E):
            self.pend_l[i] = self.pend_o[i] = None
            self.stream_l[i] = self._stream()
            self.stream_o[i] = None
            self._select_opponent(i)

    def _start_exploiter(self):
        self.frozen_main = self._freeze(self.agent.actor)
        self.exp_agent = PPOAgent(self.agent.state_dim, self.agent.num_actions,
                                  device=str(self.agent.device))
        if EXPLOITER_WARM_START:
            self.exp_agent.actor.load_state_dict(self.agent.actor.state_dict())
            self.exp_agent.critic.load_state_dict(self.agent.critic.state_dict())
        self.phase = "exploiter"
        self.phase_eps = 0
        self.exp_recent = []
        self.exp_gate_hit_at = None
        self._reset_collection()
        print(f"=== league: exploiter #{len(self.exploiters) + 1} starts "
              f"(main ep {self.main_ep_total}, "
              f"{'warm' if EXPLOITER_WARM_START else 'cold'} start) ===")

    def _finish_exploiter(self):
        wr = (sum(self.exp_recent) / len(self.exp_recent)) if self.exp_recent else 0.0
        self.exploiters.append({"net": self._freeze(self.exp_agent.actor),
                                "episode": self.episode_count, "winrate": wr,
                                "recent": []})
        if len(self.exploiters) > EXPLOITER_POOL_MAX:
            self.exploiters.pop(0)
            for j in range(self.E):
                if self.opp[j][0] == "exp":
                    self.opp[j] = ("exp", max(0, self.opp[j][1] - 1))
        how = "gated" if self.exp_gate_hit_at is not None else "timed out"
        print(f"=== league: exploiter #{len(self.exploiters)} done ({how} after "
              f"{self.phase_eps} eps) — {wr * 100:.1f}% vs main over its last "
              f"{len(self.exp_recent)} eps (pool {len(self.exploiters)}) ===")
        self.exp_agent = None
        self.frozen_main = None
        self.phase = "main"
        self.next_exploiter_at += EXPLOITER_INTERVAL
        self._reset_collection()

    # --- PPO update (same construction as the single-process trainer) --------------
    def run_update(self):
        t0 = time.perf_counter()
        rows = {k: [] for k in ("states", "actions", "logps", "values",
                                "advantages", "returns")}
        for i in range(self.E):
            for stream, pend in ((self.stream_l[i], self.pend_l[i]),
                                 (self.stream_o[i], self.pend_o[i])):
                if stream is None or not stream["s"]:
                    continue
                last_v = pend[3] if (pend is not None and not stream["d"][-1]) else 0.0
                adv, ret = compute_gae(stream["r"], stream["v"], stream["d"], last_v)
                for t in range(len(stream["s"])):
                    rows["states"].append(stream["s"][t])
                    rows["actions"].append(stream["a"][t])
                    rows["logps"].append(stream["lp"][t])
                    rows["values"].append(stream["v"][t])
                    rows["advantages"].append(adv[t])
                    rows["returns"].append(ret[t])
                    if MIRROR_AUGMENT:
                        rows["states"].append(mirror_obs(stream["s"][t]))
                        rows["actions"].append(mirror_action(stream["a"][t]))
                        rows["logps"].append(stream["lp"][t])
                        rows["values"].append(stream["v"][t])
                        rows["advantages"].append(adv[t])
                        rows["returns"].append(ret[t])
        rollout = {
            "states": np.stack(rows["states"]).astype(np.float32),
            "actions": np.array(rows["actions"], dtype=np.int64),
            "logps": np.array(rows["logps"], dtype=np.float32),
            "values": np.array(rows["values"], dtype=np.float32),
            "advantages": np.array(rows["advantages"], dtype=np.float32),
            "returns": np.array(rows["returns"], dtype=np.float32),
        }
        if self.phase == "exploiter":
            if self.exp_gate_hit_at is not None:
                ec = EXPLOITER_EC_END        # sharpening: exploit hard
            else:
                frac = min(1.0, self.phase_eps / EXPLOITER_EC_DECAY_EPISODES)
                ec = (EXPLOITER_EC_START
                      + (EXPLOITER_EC_END - EXPLOITER_EC_START) * frac)
        else:
            ec = entropy_coef_at(self.main_ep_total)
        stats = self._active_agent().update(rollout, entropy_coef=ec)
        for i in range(self.E):
            self.stream_l[i] = self._stream()
            if self.stream_o[i] is not None:
                self.stream_o[i] = self._stream()
        self.perf["train"] += time.perf_counter() - t0
        return stats, rollout["states"].shape[0]

    def win_rate(self):
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    def exp_win_rate(self):
        return (sum(self.exp_recent) / len(self.exp_recent)) if self.exp_recent else 0.0

    # --- save / load (identical format to train_ppo) --------------------------------
    def serialize(self):
        return {
            "version": 1,
            "algo": "ppo",
            "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stateDim": self.agent.state_dim,
            "progress": {
                "episodeCount": self.episode_count,
                "envSteps": self.env_steps,
                "currentRating": self.current_rating,
            },
            "agent": self.agent.serialize(),
            "snapshots": [
                {"rating": it["rating"], "episode": it["episode"],
                 "weights": it["net"].to_records()}
                for it in self.snapshots
            ],
            # League: exploiter pool + progress. An in-flight exploiter phase is
            # NOT saved (dropped on load; the phase restarts when due).
            "league": {
                "mainEpTotal": self.main_ep_total,
                "nextExploiterAt": self.next_exploiter_at,
                "exploiters": [
                    {"episode": it["episode"], "winrate": it["winrate"],
                     "weights": it["net"].to_records()}
                    for it in self.exploiters
                ],
            },
        }

    def load_state(self, obj):
        if obj.get("algo") != "ppo":
            raise ValueError(f"not a ppo save (algo={obj.get('algo')})")
        if obj.get("stateDim") != self.agent.state_dim:
            raise ValueError("stateDim mismatch")
        self.agent.load_state(obj["agent"])
        self.snapshots = []
        for rec in obj.get("snapshots", []):
            net = MLP.from_records(rec["weights"]).to(self.agent.device)
            for p in net.parameters():
                p.requires_grad_(False)
            self.snapshots.append({"net": net, "rating": rec["rating"],
                                   "episode": rec.get("episode", 0), "recent": []})
        prog = obj.get("progress", {})
        self.episode_count = prog.get("episodeCount", 0)
        self.env_steps = prog.get("envSteps", 0)
        self.current_rating = prog.get("currentRating", INITIAL_RATING)

        league = obj.get("league", {})
        self.exploiters = []
        for rec in league.get("exploiters", []):
            net = MLP.from_records(rec["weights"]).to(self.agent.device)
            for p in net.parameters():
                p.requires_grad_(False)
            self.exploiters.append({"net": net, "episode": rec.get("episode", 0),
                                    "winrate": rec.get("winrate", 0.0),
                                    "recent": []})
        self.main_ep_total = league.get("mainEpTotal", self.episode_count)
        self.next_exploiter_at = league.get(
            "nextExploiterAt",
            (self.main_ep_total // EXPLOITER_INTERVAL + 1) * EXPLOITER_INTERVAL)
        self.phase = "main"
        self.exp_agent = None
        self.frozen_main = None
        self._reset_collection()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"))
    ap.add_argument("--load")
    ap.add_argument("--out", default=str(REPO / "runs/ppo-mp"))
    ap.add_argument("--episodes", type=int, default=100_000_000)
    ap.add_argument("--save-every", type=int, default=200000)
    ap.add_argument("--replay-every", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs-per-worker", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="cap torch CPU threads (0 = leave default)")
    ap.add_argument("--exploiter-interval", type=int,
                    help="override league EXPLOITER_INTERVAL (testing)")
    ap.add_argument("--exploiter-max-episodes", type=int,
                    help="override league EXPLOITER_MAX_EPISODES (testing)")
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.exploiter_interval:
        globals()["EXPLOITER_INTERVAL"] = args.exploiter_interval
    if args.exploiter_max_episodes:
        globals()["EXPLOITER_MAX_EPISODES"] = args.exploiter_max_episodes

    out_dir = Path(args.out)
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent(C.STATE_DIM, C.NUM_ACTIONS, device=args.device)
    coll = VecCollector(args.workers, args.envs_per_worker, C.STATE_DIM,
                        args.map, C.ACTION_REPEAT, replay_dir=replay_dir,
                        replay_every=args.replay_every, tps=C.TPS,
                        input_lag=C.INPUT_LAG)
    trainer = MPPPOTrainer(coll, agent)
    print(f"PPO/mp: {args.workers} workers x {args.envs_per_worker} envs "
          f"= {coll.E}, rollout {ROLLOUT_STEPS}, device {args.device}")

    if args.load:
        with open(args.load) as f:
            trainer.load_state(json.load(f))
        print(f"resumed: episode {trainer.episode_count}, "
              f"ELO {trainer.current_rating:.1f}, snapshots {len(trainer.snapshots)}")

    def save_checkpoint(tag=""):
        path = out_dir / (f"bonk-ppo-ep{trainer.episode_count}"
                          f"-elo{round(trainer.current_rating)}{tag}.json")
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
            if trainer.learner_steps() >= ROLLOUT_STEPS:
                last_stats, _ = trainer.run_update()

            now = time.perf_counter()
            if now - last_log >= 5.0:
                p = trainer.perf
                dc = max(1, p["cycles"] - last_perf["cycles"])
                sps = (trainer.env_steps - last_steps) / (now - last_log)
                loss = (f"a={last_stats['actor_loss']:.4f} "
                        f"c={last_stats['critic_loss']:.4f} "
                        f"H={last_stats['entropy']:.3f} "
                        f"ec={last_stats['ent_coef']:.3f}") if last_stats else "warmup"
                phase = ("MAIN" if trainer.phase == "main"
                         else f"EXPL {trainer.phase_eps}/{EXPLOITER_MAX_EPISODES}"
                              f"{'*' if trainer.exp_gate_hit_at is not None else ''}"
                              f" wr {trainer.exp_win_rate()*100:.0f}%")
                print(f"ep {trainer.episode_count} [{phase}|"
                      f"snap {len(trainer.snapshots)} exp {len(trainer.exploiters)}] | "
                      f"steps/s {sps:.0f} | "
                      f"ELO {trainer.current_rating:.1f} | "
                      f"wr {trainer.win_rate()*100:.0f}% | "
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
