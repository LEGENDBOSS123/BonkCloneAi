"""Offline critic benchmark — minutes instead of hours.

The critic is doing SUPERVISED classification (prefix -> outcome), so testing it
inside the RL loop is pure waste: you pay for env stepping, actor updates and
self-play non-stationarity to measure something that only depends on the data
and the architecture.

So: collect ONE dataset of episodes from a frozen policy, then train each
candidate critic on that identical, held-out-split dataset. Every arm sees the
same data, seed variance is the only noise, and an arm takes ~1-2 min instead of
6 hours.

    python -m bonk4.critic_bench --collect 1500          # cache a dataset
    python -m bonk4.critic_bench --arms baseline,no_action_aux,no_aux,vc2,l1

What it CANNOT tell you: closed-loop effects, where a better critic changes the
advantage, which changes the policy, which changes the data. For "is this
architecture able to predict the outcome at all" and "are the aux heads stealing
gradient", it is exactly the right instrument.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C

CACHE = Path("/tmp/bonk4_critic_bench.npz")


def collect(ckpt: str, n_eps: int, spawn_frac: float, n_envs: int = 32):
    """Roll a frozen policy and store whole episodes: obs, own action, opponent
    action, outcome class."""
    from bonk3.simadapter import EngineSim

    from .env import LagEnv
    from .policy import load_policy, resolve_agent
    o = json.load(open(ckpt))
    pol = load_policy(resolve_agent(o, "main")[0])
    envs = [LagEnv(EngineSim(C.MAP_NAME)) for _ in range(n_envs)]
    for e in envs:
        e.reset(spawn_frac=spawn_frac)
    pol.reset(n_envs)
    buf = [[] for _ in range(n_envs)]
    S, A, O, L, K = [], [], [], [], []
    tick = 0
    while len(L) < n_eps:
        if tick % C.ACTION_REPEAT == 0:
            s0 = np.stack([e.decision_state(0) for e in envs])
            s1 = np.stack([e.decision_state(1) for e in envs])
            a0, a1 = pol.act(s0), pol.act(s1)
            for i, e in enumerate(envs):
                buf[i].append((s0[i], int(a0[i]), int(a1[i])))
                e.set_decision(0, int(a0[i]))
                e.set_decision(1, int(a1[i]))
        for i, e in enumerate(envs):
            r = e.tick()
            if r["done"]:
                if len(buf[i]) > 20:
                    cls = 2 if (r["timeout"] or all(r["dead"])) else (0 if r["dead"][1] else 1)
                    S.append(np.stack([b[0] for b in buf[i]]))
                    A.append(np.array([b[1] for b in buf[i]], np.int64))
                    O.append(np.array([b[2] for b in buf[i]], np.int64))
                    L.append(len(buf[i])); K.append(cls)
                buf[i] = []
                e.reset(spawn_frac=spawn_frac); pol.reset_env(i)
        tick += 1
    np.savez_compressed(CACHE, s=np.concatenate(S), a=np.concatenate(A),
                        o=np.concatenate(O), lens=np.array(L), cls=np.array(K))
    print(f"  cached {len(L)} episodes / {sum(L):,} rows -> {CACHE}")


def _pool_from(idx, d):
    from .pool import EpisodePool
    off = np.concatenate([[0], np.cumsum(d["lens"])])
    rew = np.asarray(C.OUTCOME_REWARD, np.float32)
    p = EpisodePool(int(d["lens"][idx].sum()) + 8, C.STATE_DIM)
    for i in idx:
        a, b = off[i], off[i] + d["lens"][i]
        c = int(d["cls"][i])
        p.add(d["s"][a:b], d["a"][a:b], np.zeros(b - a, np.float32),
              np.zeros(b - a, np.float32), d["o"][a:b], d["a"][a:b],
              c, float(rew[c]))
    return p


ARMS = {
    "baseline":       dict(),
    # Speed arms. The critic is ~85% of update time and every measurement so
    # far says it is information- not capacity-limited (ev plateaus ~0.10
    # whatever we do), so the question is whether a smaller one loses anything.
    "l1":             dict(CRITIC_LAYERS=1),
    "h128":           dict(CRITIC_HIDDEN=128),
    "l1_h128":        dict(CRITIC_LAYERS=1, CRITIC_HIDDEN=128),
    "no_action_aux":  dict(AUX_OPP_ACTION_COEF=0.0, AUX_SELF_ACTION_COEF=0.0),
    "no_aux":         dict(AUX_OPP_ACTION_COEF=0.0, AUX_SELF_ACTION_COEF=0.0,
                           AUX_NEXT_POS_COEF=0.0),
    "vc2":            dict(AUX_OPP_ACTION_COEF=0.0, AUX_SELF_ACTION_COEF=0.0,
                           VALUE_COEF=2.0),
    "l1":             dict(CRITIC_LAYERS=1),
    "l1_no_aux":      dict(CRITIC_LAYERS=1, AUX_OPP_ACTION_COEF=0.0,
                           AUX_SELF_ACTION_COEF=0.0, AUX_NEXT_POS_COEF=0.0),
}


def evaluate(agent, pool, D):
    """Held-out ev, overall and over the last 20% of each episode (the only
    region where the outcome is knowable at all — see the decile analysis)."""
    from .pool import critic_batches
    rew = np.asarray(C.OUTCOME_REWARD, np.float32)
    Vs, Rs, late_V, late_R = [], [], [], []
    agent.critic.eval()
    with torch.no_grad():
        for bt in critic_batches(pool, C.CRITIC_LENGTH_BUCKETS, D):
            obs = torch.as_tensor(bt["states"])
            m = bt["target_mask"]
            z = agent.actor.trunk(obs)
            lg, _f, _h = agent.critic.unroll(
                z, agent.critic.initial_state(obs.shape[1], agent.device),
                torch.zeros(obs.shape[0], obs.shape[1]))
            v = agent.critic.value_of(lg).numpy()
            r = (bt["targets"] * rew).sum(-1)
            Vs.append(v[m]); Rs.append(r[m])
            T = obs.shape[0]
            cut = np.zeros_like(m)
            lens = m.sum(0)
            for j, L in enumerate(lens):
                cut[max(0, int(L * 0.8)):int(L), j] = True
            late_V.append(v[cut]); late_R.append(r[cut])
    agent.critic.train()
    V, R = np.concatenate(Vs), np.concatenate(Rs)
    lV, lR = np.concatenate(late_V), np.concatenate(late_R)
    return (1 - (R - V).var() / max(R.var(), 1e-9),
            1 - (lR - lV).var() / max(lR.var(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../runs/ppo4/bonk4-ppo-ep4113159-elo1685-final.json")
    ap.add_argument("--collect", type=int, default=0)
    ap.add_argument("--spawn-frac", type=float, default=0.4)
    ap.add_argument("--arms", default="baseline,no_action_aux,no_aux,vc2,l1_no_aux")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.collect:
        collect(args.ckpt, args.collect, args.spawn_frac)
        return
    if not CACHE.exists():
        raise SystemExit(f"no dataset at {CACHE}; run with --collect 1500 first")

    d = dict(np.load(CACHE))
    n = len(d["lens"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr_i, va_i = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    print(f"  dataset {n} episodes ({d['lens'].sum():,} rows) | "
          f"train {len(tr_i)} val {len(va_i)} | "
          f"classes {dict(zip(*np.unique(d['cls'], return_counts=True)))}")
    tr, va = _pool_from(tr_i, d), _pool_from(va_i, d)

    from .ppo import PPOAgent
    from .pool import critic_batches
    base = {k: getattr(C, k) for k in
            ("AUX_OPP_ACTION_COEF", "AUX_SELF_ACTION_COEF", "AUX_NEXT_POS_COEF",
             "VALUE_COEF", "CRITIC_LAYERS", "CRITIC_EPOCHS", "CRITIC_HIDDEN")}
    print(f"\n  {'arm':<14} {'val ev':>8} {'ev last20%':>11} {'train c':>9} {'s':>6}")
    for name in args.arms.split(","):
        for k, v in base.items():
            setattr(C, k, v)
        for k, v in ARMS[name].items():
            setattr(C, k, v)
        C.CRITIC_EPOCHS = 1
        torch.manual_seed(args.seed)
        agent = PPOAgent(C.STATE_DIM, C.NUM_ACTIONS, "cpu")
        t0 = time.perf_counter()
        batches = list(critic_batches(tr, C.CRITIC_LENGTH_BUCKETS, C.STATE_DIM))
        st = {}
        for _ in range(args.epochs):
            st = agent.update_critic_episodes(batches)
        ev, ev_late = evaluate(agent, va, C.STATE_DIM)
        print(f"  {name:<14} {ev:>+8.3f} {ev_late:>+11.3f} "
              f"{st['critic_loss']:>9.3f} {time.perf_counter()-t0:>6.0f}")
    for k, v in base.items():
        setattr(C, k, v)


if __name__ == "__main__":
    main()
