"""Self-play PPO training (the rl1 recipe, PyTorch speed).

Structure mirrors src/rl/trainer.mjs: N parallel LagEnvs; the learner is seat 0
of every env; seat 1 is the live model (prob OPPONENT_CURRENT_PROB, its
trajectory also trains — opponent-POV augmentation) or a frozen actor snapshot
(rated: ELO). Global decision alignment every ACTION_REPEAT ticks, rewards
accumulate per decision block, GAE per stream, horizontal-mirror augmentation.

  python -m bonk.train_ppo [--load ckpt.json] [--out ../runs/ppo] ...

Checkpoints: rl2-style JSON with algo "ppo"; `agent.actor` records are what a
future play2.mjs (or play.py --net actor) consumes. Ctrl-C saves and exits.
"""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from . import config as C
from .env2 import LagEnv
from .networks import MLP
from .ppo import PPOAgent, HIDDEN, entropy_coef_at
from .sim import BonkSim
from .train import ReplayRecorder
from .trainer import elo_update, mirror_action, mirror_obs

REPO = Path(__file__).resolve().parents[2]

NUM_ENVS = 256
TOTAL_EPISODES = 500_000
GAMMA = 0.99          # per decision (ACTION_REPEAT ticks)
GAE_LAMBDA = 0.95
ROLLOUT_STEPS = 32_768   # learner decisions per PPO update
MIRROR_AUGMENT = True
OPPONENT_CURRENT_PROB = 0.5
SNAPSHOT_INTERVAL = 200_000  # episodes
SNAPSHOT_BUFFER = 50
INITIAL_RATING = 1000.0


def compute_gae(rewards, values, dones, last_value):
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    ret = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        next_v = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * next_v * nonterminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * nonterminal * gae
        adv[t] = gae
        ret[t] = gae + values[t]
    return adv, ret


class PPOTrainer:
    def __init__(self, envs, agent: PPOAgent):
        self.envs = envs
        self.agent = agent
        self.K = C.ACTION_REPEAT

        self.snapshots = []  # {net, rating, episode}
        self.current_rating = INITIAL_RATING
        self.episode_count = 0
        self.tick_count = 0
        self.env_steps = 0
        self.recent = []  # learner scores for win-rate
        self.recorder = None
        self.perf = {"decide": 0.0, "phys": 0.0, "train": 0.0, "ticks": 0}

        self.slots = []
        for env in envs:
            env.reset()
            slot = {"env": env, "opponent": None,
                    "stream_l": self._stream(), "stream_o": None,
                    "pend_l": None, "pend_o": None}
            self.slots.append(slot)
            self._select_opponent(slot)

    @staticmethod
    def _stream():
        return {"s": [], "a": [], "lp": [], "v": [], "r": [], "d": []}

    def _select_opponent(self, slot):
        if not self.snapshots or np.random.random() < OPPONENT_CURRENT_PROB:
            slot["opponent"] = ("current", None)
            slot["stream_o"] = self._stream()
        else:
            slot["opponent"] = ("snap", np.random.randint(len(self.snapshots)))
            slot["stream_o"] = None

    def learner_steps(self):
        return sum(len(s["stream_l"]["s"]) for s in self.slots)

    # --- one vectorized tick ---------------------------------------------------
    def tick(self):
        t0 = time.perf_counter()
        infos = []
        if self.tick_count % self.K == 0:
            l_states, o_cur, o_snap = [], [], {}
            for slot in self.slots:
                st = slot["env"].decision_state(0)
                if slot["pend_l"] is not None:
                    self._complete(slot["stream_l"], slot["pend_l"], False)
                l_states.append((slot, st))

                ost = slot["env"].decision_state(1)
                kind, snap_i = slot["opponent"]
                if kind == "current":
                    if slot["pend_o"] is not None:
                        self._complete(slot["stream_o"], slot["pend_o"], False)
                    o_cur.append((slot, ost))
                else:
                    o_snap.setdefault(snap_i, []).append((slot, ost))

            acts, lps, vals = self.agent.act_batch(np.stack([s for _, s in l_states]))
            for (slot, st), a, lp, v in zip(l_states, acts, lps, vals):
                slot["pend_l"] = [st, int(a), float(lp), float(v), 0.0]
                slot["env"].set_decision(0, int(a))
            if o_cur:
                acts, lps, vals = self.agent.act_batch(np.stack([s for _, s in o_cur]))
                for (slot, st), a, lp, v in zip(o_cur, acts, lps, vals):
                    slot["pend_o"] = [st, int(a), float(lp), float(v), 0.0]
                    slot["env"].set_decision(1, int(a))
            for snap_i, reqs in o_snap.items():
                net = self.snapshots[snap_i]["net"]
                acts = self.agent.act_actions(net, np.stack([s for _, s in reqs]))
                for (slot, _), a in zip(reqs, acts):
                    slot["env"].set_decision(1, int(a))

        t1 = time.perf_counter()
        for slot in self.slots:
            res = slot["env"].tick()
            self.env_steps += 1
            if slot["pend_l"] is not None:
                slot["pend_l"][4] += res["rewards"][0]
            if slot["pend_o"] is not None:
                slot["pend_o"][4] += res["rewards"][1]
            if self.recorder and slot is self.slots[0]:
                self.recorder.frame(slot["env"], res)
            if res["done"]:
                infos.append(self._finish_episode(slot, res))
        self.tick_count += 1
        self.perf["decide"] += t1 - t0
        self.perf["phys"] += time.perf_counter() - t1
        self.perf["ticks"] += 1
        return infos

    @staticmethod
    def _complete(stream, pend, done):
        s, a, lp, v, r = pend
        stream["s"].append(s)
        stream["a"].append(a)
        stream["lp"].append(lp)
        stream["v"].append(v)
        stream["r"].append(r)
        stream["d"].append(1.0 if done else 0.0)

    def _finish_episode(self, slot, res):
        if slot["pend_l"] is not None:
            self._complete(slot["stream_l"], slot["pend_l"], True)
        if slot["pend_o"] is not None:
            self._complete(slot["stream_o"], slot["pend_o"], True)
        slot["pend_l"] = slot["pend_o"] = None

        if res["timeout"] or (res["dead"][0] and res["dead"][1]):
            score = 0.5
        elif res["dead"][1]:
            score = 1.0
        else:
            score = 0.0
        self.recent.append(score)
        del self.recent[:-100]

        kind, snap_i = slot["opponent"]
        if kind == "snap":
            item = self.snapshots[snap_i]
            self.current_rating, item["rating"] = elo_update(
                self.current_rating, item["rating"], score)

        self.episode_count += 1
        if self.episode_count % SNAPSHOT_INTERVAL == 0:
            net = MLP(self.agent.state_dim, HIDDEN, self.agent.num_actions)
            net.load_state_dict(self.agent.actor.state_dict())
            net.to(self.agent.device)
            for p in net.parameters():
                p.requires_grad_(False)
            self.snapshots.append({"net": net, "rating": self.current_rating,
                                   "episode": self.episode_count})
            if len(self.snapshots) > SNAPSHOT_BUFFER:
                self.snapshots.pop(0)

        slot["env"].reset()
        self._select_opponent(slot)
        return {"episode": self.episode_count, "score": score}

    # --- PPO update ------------------------------------------------------------------
    def run_update(self):
        t0 = time.perf_counter()
        rows = {k: [] for k in ("states", "actions", "logps", "values",
                                "advantages", "returns")}
        for slot in self.slots:
            for stream, pend in ((slot["stream_l"], slot["pend_l"]),
                                 (slot["stream_o"], slot["pend_o"])):
                if stream is None or not stream["s"]:
                    continue
                # Bootstrap: mid-episode streams use the in-flight decision's
                # value (it IS V(next decision state)); finished streams use 0.
                last_v = pend[3] if (pend is not None and not stream["d"][-1]) else 0.0
                adv, ret = compute_gae(stream["r"], stream["v"], stream["d"], last_v)
                for i in range(len(stream["s"])):
                    rows["states"].append(stream["s"][i])
                    rows["actions"].append(stream["a"][i])
                    rows["logps"].append(stream["lp"][i])
                    rows["values"].append(stream["v"][i])
                    rows["advantages"].append(adv[i])
                    rows["returns"].append(ret[i])
                    if MIRROR_AUGMENT:
                        rows["states"].append(mirror_obs(stream["s"][i]))
                        rows["actions"].append(mirror_action(stream["a"][i]))
                        rows["logps"].append(stream["lp"][i])
                        rows["values"].append(stream["v"][i])
                        rows["advantages"].append(adv[i])
                        rows["returns"].append(ret[i])

        rollout = {
            "states": np.stack(rows["states"]).astype(np.float32),
            "actions": np.array(rows["actions"], dtype=np.int64),
            "logps": np.array(rows["logps"], dtype=np.float32),
            "values": np.array(rows["values"], dtype=np.float32),
            "advantages": np.array(rows["advantages"], dtype=np.float32),
            "returns": np.array(rows["returns"], dtype=np.float32),
        }
        stats = self.agent.update(
            rollout, entropy_coef=entropy_coef_at(self.episode_count))
        for slot in self.slots:
            slot["stream_l"] = self._stream()
            if slot["stream_o"] is not None:
                slot["stream_o"] = self._stream()
        self.perf["train"] += time.perf_counter() - t0
        return stats, rollout["states"].shape[0]

    def win_rate(self):
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    # --- save / load ---------------------------------------------------------------------
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
                                   "episode": rec.get("episode", 0)})
        prog = obj.get("progress", {})
        self.episode_count = prog.get("episodeCount", 0)
        self.env_steps = prog.get("envSteps", 0)
        self.current_rating = prog.get("currentRating", INITIAL_RATING)
        for slot in self.slots:
            slot["env"].reset()
            slot["pend_l"] = slot["pend_o"] = None
            slot["stream_l"] = self._stream()
            self._select_opponent(slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"))
    ap.add_argument("--load")
    ap.add_argument("--out", default=str(REPO / "runs/ppo"))
    ap.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--replay-every", type=int, default=20000)
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "replays").mkdir(parents=True, exist_ok=True)
    with open(args.map) as f:
        map_json = json.load(f)
    envs = [LagEnv(BonkSim(map_json)) for _ in range(args.num_envs)]
    agent = PPOAgent(envs[0].state_dim, C.NUM_ACTIONS, device=args.device)
    trainer = PPOTrainer(envs, agent)
    trainer.recorder = ReplayRecorder(out_dir / "replays", args.replay_every, trainer)
    print(f"PPO/pytorch: {args.num_envs} envs, obs {envs[0].state_dim}, "
          f"rollout {ROLLOUT_STEPS}, device {args.device}")

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

    while trainer.episode_count < args.episodes and not stop["flag"]:
        trainer.tick()
        if trainer.learner_steps() >= ROLLOUT_STEPS:
            last_stats, rows = trainer.run_update()

        now = time.perf_counter()
        if now - last_log >= 5.0:
            p = trainer.perf
            dt = max(1, p["ticks"] - last_perf["ticks"])
            sps = (trainer.env_steps - last_steps) / (now - last_log)
            loss = (f"a={last_stats['actor_loss']:.4f} c={last_stats['critic_loss']:.4f} "
                    f"H={last_stats['entropy']:.3f} "
                    f"ec={last_stats['ent_coef']:.3f}") if last_stats else "warmup"
            print(f"ep {trainer.episode_count} | steps/s {sps:.0f} | "
                  f"ELO {trainer.current_rating:.1f} | wr {trainer.win_rate()*100:.0f}% | "
                  f"updates {agent.updates} | {loss} | "
                  f"perf decide={(p['decide']-last_perf['decide'])/dt*1000:.2f} "
                  f"phys={(p['phys']-last_perf['phys'])/dt*1000:.2f} "
                  f"train={(p['train']-last_perf['train'])/dt*1000:.2f}ms/tick")
            last_log, last_steps, last_perf = now, trainer.env_steps, dict(p)

        if trainer.episode_count - last_saved >= args.save_every:
            last_saved = trainer.episode_count
            save_checkpoint()

    save_checkpoint("-final")
    print(f"done: {trainer.episode_count} episodes in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
