"""Vectorized NFSP self-play trainer — port of src/rl2/trainer.mjs.

Same structure: N parallel LagEnvs, global decision alignment every
ACTION_REPEAT ticks, n-step transitions into the replay, BR (state, action)
pairs into the reservoir, mirror augmentation, eval episodes vs frozen AVG
snapshots (ELO), interleaved gradient steps. Checkpoints use the exact rl2
browser JSON layout (version/algo/stateDim/progress/agent/snapshots).
"""

import math
import random
import time

import numpy as np

from . import config as C
from .buffers import ReplayBuffer, ReservoirBuffer
from .networks import MLP
from .nfsp import NFSPAgent, AVG_BATCH, SAC_BATCH, HIDDEN

# NFSP / eval settings mirroring src/rl2/config.mjs.
ANTICIPATORY = 0.15
REPLAY_CAPACITY = 300_000
RESERVOIR_CAPACITY = 500_000
LEARN_START = 5_000
TRAIN_EVERY_TICKS = 8
SAC_STEPS = 1
SL_STEPS = 1
MIRROR_AUGMENT = True
GAMMA = 0.997
N_STEP = 3
EVAL_PROB = 0.06
SNAPSHOT_INTERVAL = 500
SNAPSHOT_BUFFER = 30
INITIAL_RATING = 1000.0
K_FACTOR = 16.0

# Horizontal mirror of the 30-dim obs (see lag_env.mjs).
MIRROR_NEGATE = [0, 2, 10, 12, 20, 22]
MIRROR_SWAPS = [(7, 8), (17, 18), (26, 27)]


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    out[MIRROR_NEGATE] *= -1
    for a, b in MIRROR_SWAPS:
        out[a], out[b] = out[b], out[a]
    return out


def mirror_action(index: int) -> int:
    lr, rest = index // 6, index % 6
    return (2 if lr == 1 else 1 if lr == 2 else 0) * 6 + rest


def elo_update(ra, rb, score_a, k=K_FACTOR):
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


class Trainer:
    def __init__(self, envs, agent: NFSPAgent):
        self.envs = envs
        self.agent = agent
        self.K = C.ACTION_REPEAT
        self.replay = ReplayBuffer(REPLAY_CAPACITY, agent.state_dim)
        self.reservoir = ReservoirBuffer(RESERVOIR_CAPACITY, agent.state_dim)

        self.snapshots = []  # {net, rating, episode}
        self.current_rating = INITIAL_RATING
        self.episode_count = 0
        self.tick_count = 0
        self.env_steps = 0
        self.eval_results = []
        self.br_results = []
        self.recorder = None
        self.perf = {"decide": 0.0, "phys": 0.0, "train": 0.0, "ticks": 0}

        self.slots = []
        for env in envs:
            env.reset()
            slot = {"env": env, "pending": [None, None], "nq": [[], []],
                    "modes": None, "is_eval": False, "eval_seat": 0}
            self.slots.append(slot)
            self._select_modes(slot)

    # --- opponents / modes ---------------------------------------------------
    def _select_modes(self, slot):
        if self.snapshots and random.random() < EVAL_PROB:
            slot["is_eval"] = True
            slot["eval_seat"] = random.randint(0, 1)
            item = random.choice(self.snapshots)
            slot["modes"] = [None, None]
            slot["modes"][slot["eval_seat"]] = ("avg", None)
            slot["modes"][1 - slot["eval_seat"]] = ("snap", item)
        else:
            slot["is_eval"] = False
            slot["modes"] = [
                ("br" if random.random() < ANTICIPATORY else "avg", None)
                for _ in range(2)
            ]

    # --- n-step assembly ---------------------------------------------------------
    def _push_step(self, slot, seat, s, a, r, s2, done):
        q = slot["nq"][seat]
        q.append((s, a, r))
        if done:
            for j in range(len(q)):
                g = 0.0
                for i in range(len(q) - 1, j - 1, -1):
                    g = q[i][2] + GAMMA * g
                self._emit(q[j][0], q[j][1], g, s2, True, GAMMA ** (len(q) - j))
            q.clear()
        elif len(q) >= N_STEP:
            g = 0.0
            for i in range(len(q) - 1, -1, -1):
                g = q[i][2] + GAMMA * g
            self._emit(q[0][0], q[0][1], g, s2, False, GAMMA ** len(q))
            q.pop(0)

    def _emit(self, s, a, r, s2, done, gamma_n):
        self.replay.push(s, a, r, s2, done, gamma_n)
        if MIRROR_AUGMENT:
            self.replay.push(mirror_obs(s), mirror_action(a), r,
                             mirror_obs(s2), done, gamma_n)

    # --- one vectorized tick ---------------------------------------------------------
    def tick(self):
        t0 = time.perf_counter()
        infos = []
        if self.tick_count % self.K == 0:
            deciding = []  # (slot, seat, state)
            for slot in self.slots:
                for seat in (0, 1):
                    state = slot["env"].decision_state(seat)
                    p = slot["pending"][seat]
                    if p is not None:
                        self._push_step(slot, seat, p[0], p[1], p[2], state, False)
                    deciding.append((slot, seat, state))

            groups = {}
            for req in deciding:
                mode = req[0]["modes"][req[1]]
                key = id(mode[1]["net"]) if mode[0] == "snap" else mode[0]
                groups.setdefault(key, ([], None))
                groups[key][0].append(req)
                if mode[0] == "snap":
                    groups[key] = (groups[key][0], mode[1]["net"])
            for key, (reqs, snap_net) in groups.items():
                net = (self.agent.policy if key == "br"
                       else self.agent.avg if key == "avg" else snap_net)
                states = np.stack([r[2] for r in reqs])
                actions = self.agent.act_batch(net, states, greedy=False)
                for (slot, seat, state), a in zip(reqs, actions):
                    a = int(a)
                    slot["pending"][seat] = [state, a, 0.0]
                    slot["env"].set_decision(seat, a)
                    if not slot["is_eval"] and slot["modes"][seat][0] == "br":
                        self.reservoir.push(state, a)
                        if MIRROR_AUGMENT:
                            self.reservoir.push(mirror_obs(state), mirror_action(a))

        t1 = time.perf_counter()
        for slot in self.slots:
            res = slot["env"].tick()
            self.env_steps += 1
            for seat in (0, 1):
                if slot["pending"][seat] is not None:
                    slot["pending"][seat][2] += res["rewards"][seat]
            if self.recorder and slot is self.slots[0]:
                self.recorder.frame(slot["env"], res)
            if res["done"]:
                infos.append(self._finish_episode(slot, res))
        self.tick_count += 1
        self.perf["decide"] += t1 - t0
        self.perf["phys"] += time.perf_counter() - t1
        self.perf["ticks"] += 1
        return infos

    def _finish_episode(self, slot, res):
        for seat in (0, 1):
            p = slot["pending"][seat]
            if p is not None:
                self._push_step(slot, seat, p[0], p[1], p[2], p[0], True)
            slot["pending"][seat] = None
            slot["nq"][seat].clear()

        def score_for(seat):
            if res["timeout"] or (res["dead"][0] and res["dead"][1]):
                return 0.5
            return 1.0 if (res["dead"][1 - seat] and not res["dead"][seat]) else 0.0

        if slot["is_eval"]:
            score = score_for(slot["eval_seat"])
            item = slot["modes"][1 - slot["eval_seat"]][1]
            self.current_rating, item["rating"] = elo_update(
                self.current_rating, item["rating"], score)
            self.eval_results.append(score)
            del self.eval_results[:-200]
        else:
            kinds = [slot["modes"][0][0], slot["modes"][1][0]]
            if "br" in kinds and "avg" in kinds:
                self.br_results.append(score_for(kinds.index("br")))
                del self.br_results[:-300]

        self.episode_count += 1
        if self.episode_count % SNAPSHOT_INTERVAL == 0:
            net = MLP(self.agent.state_dim, self.agent.avg.hidden, self.agent.num_actions)
            net.load_state_dict(self.agent.avg.state_dict())
            net.to(self.agent.device)
            for p in net.parameters():
                p.requires_grad_(False)
            self.snapshots.append({"net": net, "rating": self.current_rating,
                                   "episode": self.episode_count})
            if len(self.snapshots) > SNAPSHOT_BUFFER:
                self.snapshots.pop(0)

        slot["env"].reset()
        self._select_modes(slot)
        return {"episode": self.episode_count, "rating": self.current_rating}

    # --- interleaved learning ------------------------------------------------------------
    def train_tick(self):
        if self.replay.size < LEARN_START:
            return
        if self.tick_count % TRAIN_EVERY_TICKS != 2:
            return
        t0 = time.perf_counter()
        for _ in range(SAC_STEPS):
            self.agent.sac_step(self.replay.sample(SAC_BATCH))
        if self.reservoir.size >= AVG_BATCH:
            for _ in range(SL_STEPS):
                self.agent.sl_step(self.reservoir.sample(AVG_BATCH))
        self.perf["train"] += time.perf_counter() - t0

    # --- monitoring -----------------------------------------------------------------------
    def eval_win_rate(self):
        return (sum(self.eval_results) / len(self.eval_results)) if self.eval_results else 0.0

    def br_win_rate(self):
        return (sum(self.br_results) / len(self.br_results)) if self.br_results else 0.5

    # --- rl2-format save / load (browser-compatible) ----------------------------------------
    def serialize(self) -> dict:
        return {
            "version": 1,
            "algo": "nfsp-sac",
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

    def load_state(self, obj: dict):
        if obj.get("algo") != "nfsp-sac":
            raise ValueError(f"not an rl2 save (algo={obj.get('algo')})")
        if obj.get("stateDim") != self.agent.state_dim:
            raise ValueError(
                f"save stateDim {obj.get('stateDim')} != current {self.agent.state_dim}")
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
            slot["pending"] = [None, None]
            slot["nq"] = [[], []]
            self._select_modes(slot)
