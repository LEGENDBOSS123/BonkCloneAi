"""Pre-train the movement CORE by RL on a dense goal-reaching reward.

    python -m bonk4.pretrain_move --iters 300 --out ../runs/move_core.json

Why RL and not cloning: hindsight behaviour cloning on self-play data reached
10.1% action accuracy against a 6.0% baseline, i.e. nothing. The premise of
hindsight relabelling is that the action taken was a good way of reaching where
you ended up, and for a random walker that is false — a_t is uncorrelated with
where it drifted 4-40 steps later, so the labels are noise. Optimising the dense
reward directly does not care how good the demonstrator is.

The task: single agent, no opponent, random reachable goal, reward

    r_t = -( ||pos - goal_pos|| / POS_W  +  VEL_W * ||vel - goal_vel|| / POS_W )

dense at every step, plus a penalty for dying. Purely a motor objective — "get
to this point at this speed" needs no knowledge of bonk, so it satisfies the
project's constraint.

Only the CORE is kept; l_in / l_out / v_out are scaffolding, discarded when the
actor learns its own projection into the CORE's embedding space.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bonk3.simadapter import EngineSim

from . import config as C
from .env import LagEnv
from .movement import GOAL_DIM, OWN_DIM, MovementNet, goal_reward

# obs[0:4] is position/velocity ALREADY SCALED by POS_SCALE / VEL_SCALE, so the
# goal must be scaled the same way before it reaches the net. Feeding raw metres
# meant the network saw own-position ~0.5 against a goal of ~36 -- two
# quantities 30x apart through a LayerNorm -- and simply could not relate them.
GOAL_SCALE = np.array([C.POS_SCALE, C.POS_SCALE, C.VEL_SCALE, C.VEL_SCALE],
                      dtype=np.float32)


def sample_goals(sims, rng, visited=None):
    """A target near where the agent currently is.

    Sampled from states the agent has ACTUALLY OCCUPIED when a history is
    available, because a randomly offset point on this map usually lands over
    the death ring. An unreachable goal makes standing still optimal -- you
    cannot arrive, and trying costs you the death penalty -- which is exactly
    the degenerate attractor the whole module exists to escape. Reusing visited
    states guarantees somebody stood there.
    """
    g = np.zeros((len(sims), GOAL_DIM), dtype=np.float32)
    r = C.MOVE_GOAL_RADIUS
    for i, s in enumerate(sims):
        px, py = s.players[0].pos
        cand = None
        if visited is not None and len(visited) > 64:
            V = visited[rng.integers(0, len(visited), 32)]
            d = np.hypot(V[:, 0] - px, V[:, 1] - py)
            near = V[(d > 1.5) & (d < r)]
            if len(near):
                cand = near[rng.integers(0, len(near))]
        if cand is None:                       # fall back to a random offset
            g[i] = (px + rng.uniform(-r, r), py + rng.uniform(-r * .5, r * .5),
                    rng.uniform(-4, 4), rng.uniform(-2, 2))
        else:
            g[i] = cand
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--arrive", type=float, default=1.0,
                    help="bonus per step while inside ARRIVE_M of the goal")
    ap.add_argument("--arrive-m", type=float, default=1.5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2]
                                        / "runs" / "move_core.json"))
    ap.add_argument("--map", default=None, help="override C.MAP_NAME")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    E, T = args.envs, C.MOVE_EP_DECISIONS
    mp_name = args.map or C.MAP_NAME
    print(f"  map: {mp_name}")
    envs = [LagEnv(EngineSim(mp_name)) for _ in range(E)]
    for e in envs:
        e.reset(spawn_frac=1.0)
    visited = np.zeros((0, GOAL_DIM), np.float32)
    net = MovementNet(C.NUM_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"  movement pre-train | E={E} ep={T} decisions | CORE "
          f"{C.MOVE_EMB}->{C.MOVE_HIDDEN}->{C.MOVE_EMB} "
          f"({sum(p.numel() for p in net.core.parameters()):,} params)")
    print(f"  reward = -(pos_err/{C.MOVE_POS_W} + {C.MOVE_VEL_W}*vel_err/{C.MOVE_POS_W})\n")
    print(f"  {'iter':>5} {'final dist':>11} {'start dist':>11} {'closed':>8} "
          f"{'entropy':>8} {'alive':>6} {'s':>5}")

    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        for e in envs:
            e.reset(spawn_frac=1.0)
        goals = sample_goals([e.sim for e in envs], rng, visited)
        d0 = np.array([np.linalg.norm(np.array(e.sim.players[0].pos) - goals[i, :2])
                       for i, e in enumerate(envs)])
        O = np.zeros((T, E, OWN_DIM + GOAL_DIM), np.float32)
        A = np.zeros((T, E), np.int64)
        LP = np.zeros((T, E), np.float32)
        V = np.zeros((T, E), np.float32)
        R = np.zeros((T, E), np.float32)
        D = np.zeros((T, E), np.float32)
        dead = np.zeros(E, bool)
        # A dead disc is ZEROED IN PLACE by the engine, so its reported position
        # jumps ~40 m. Freezing the last living position keeps that out of both
        # the reward and the metric -- otherwise death is punished by a huge
        # phantom distance on top of the intended penalty, which teaches
        # extreme conservatism (i.e. standing still) all over again.
        last_p = np.zeros((E, 2), np.float32)
        last_v = np.zeros((E, 2), np.float32)
        for i, e in enumerate(envs):
            last_p[i] = e.sim.players[0].pos
            last_v[i] = e.sim.players[0].vel

        for t in range(T):
            own = np.stack([e.decision_state(0)[:OWN_DIM] for e in envs])
            x = np.concatenate([own, goals * GOAL_SCALE], axis=1).astype(np.float32)
            with torch.no_grad():
                gs = (goals * GOAL_SCALE).astype(np.float32)
                lg, c = net(torch.from_numpy(own), torch.from_numpy(gs))
                lp_all = F.log_softmax(lg, -1)
                a = torch.multinomial(lp_all.exp(), 1).squeeze(1)
                O[t] = x
                A[t] = a.numpy()
                LP[t] = lp_all.gather(1, a.unsqueeze(1)).squeeze(1).numpy()
                V[t] = net.v_out(c).squeeze(-1).numpy()
            for i, e in enumerate(envs):
                e.set_decision(0, int(a[i]))
                e.set_decision(1, 0)                # opponent idles
            for _ in range(C.ACTION_REPEAT):
                for i, e in enumerate(envs):
                    res = e.tick()
                    if res["done"] and not dead[i]:
                        dead[i] = True
            seen = np.array([[*e.sim.players[0].pos, *e.sim.players[0].vel]
                             for e in envs], np.float32)
            visited = np.concatenate([visited, seen])[-40000:]
            for i, e in enumerate(envs):
                if not dead[i]:
                    last_p[i] = e.sim.players[0].pos
                    last_v[i] = e.sim.players[0].vel
                if dead[i]:
                    # one-off penalty at the moment of death, nothing after:
                    # charging it every remaining step made death dominate the
                    # distance term and taught "never move" all over again.
                    R[t, i] = -C.MOVE_DEATH_PENALTY if D[max(t - 1, 0), i] == 0 \
                        and not (t > 0 and D[t - 1, i] > 0) else 0.0
                    D[t, i] = 1.0
                else:
                    R[t, i] = goal_reward(last_p[i], last_v[i], goals[i])
                    # Pure negative distance flattens out near the goal, so the
                    # entropy bonus wins and the policy drifts back to "hold
                    # still". An explicit arrival bonus keeps a gradient where
                    # it matters.
                    if np.linalg.norm(last_p[i] - goals[i, :2]) < args.arrive_m:
                        R[t, i] += args.arrive

        # GAE over the fixed-length rollout
        adv = np.zeros_like(R)
        gae = np.zeros(E, np.float32)
        for t in range(T - 1, -1, -1):
            nv = V[t + 1] if t + 1 < T else 0.0
            nt = 1.0 - D[t]
            delta = R[t] + 0.99 * nv * nt - V[t]
            gae = delta + 0.99 * 0.95 * nt * gae
            adv[t] = gae
        ret = adv + V
        n = T * E
        ob = torch.from_numpy(O.reshape(n, -1))
        ac = torch.from_numpy(A.reshape(n))
        olp = torch.from_numpy(LP.reshape(n))
        rt = torch.from_numpy(ret.reshape(n))
        ad = adv.reshape(n)
        ad = torch.from_numpy((ad - ad.mean()) / (ad.std() + 1e-8))

        ent_v = 0.0
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, 4096):
                idx = perm[s:s + 4096]
                lg, c = net(ob[idx, :OWN_DIM], ob[idx, OWN_DIM:])
                lpa = F.log_softmax(lg, -1)
                lp = lpa.gather(1, ac[idx].unsqueeze(1)).squeeze(1)
                ratio = (lp - olp[idx]).exp()
                s1 = ratio * ad[idx]
                s2 = ratio.clamp(0.8, 1.2) * ad[idx]
                ent = -(lpa.exp() * lpa).sum(-1).mean()
                v = net.v_out(c).squeeze(-1)
                loss = -torch.min(s1, s2).mean() - args.ent * ent \
                    + 0.5 * F.mse_loss(v, rt[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                ent_v = ent.item()

        if it % 10 == 0 or it == 1:
            df = np.linalg.norm(last_p - goals[:, :2], axis=1)
            alive = ~dead
            a_df, a_d0 = df[alive], d0[alive]
            print(f"  {it:>5} {a_df.mean():>10.2f}m {a_d0.mean():>10.2f}m "
                  f"{(a_d0 - a_df).mean():>7.2f}m {ent_v:>8.3f} "
                  f"{alive.mean()*100:>5.0f}% {time.perf_counter() - t0:>5.0f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"format": "move-core-v1", "d_emb": C.MOVE_EMB,
                   "hidden": C.MOVE_HIDDEN,
                   "emb_mean": net.emb_mean.tolist(),
                   "emb_std": net.emb_var.sqrt().tolist(),
                   "core": {k: v.tolist()
                            for k, v in net.core.state_dict().items()}}, f)
    print(f"\n  CORE saved -> {args.out}")


if __name__ == "__main__":
    main()
