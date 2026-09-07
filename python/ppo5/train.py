"""ppo5 — run ppo4's EXACT algorithm on Slime Volleyball.

    python -m ppo5.train --envs 64 --updates 40

Why this exists: on DEATH the critic reaches ev = 0.011 after 4M episodes, so
the actor's advantage is ~pure "did this game win" spread over ~350 actions.
Two explanations fit — the task is a brutally hard credit-assignment problem,
or something in the algorithm is broken. This separates them.

It does NOT reimplement the algorithm. It overrides bonk4.config for the new
observation/action shape, then imports bonk4's own PPOAgent (shared encoder +
prefix-GRU 3-class critic), EpisodePool, critic_batches and actor_rows. A bug in
that path shows up here identically.

Read `ev`. If it climbs well above the 0.011 seen on DEATH, the machinery is
sound and DEATH is simply hard. If it stays pinned near zero on a known-good
self-play benchmark, the bug is in the algorithm and no reward tuning will help.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import env as SV


def configure(args):
    """Point bonk4.config at Slime Volleyball before the algorithm is built."""
    from bonk4 import config as C
    C.STATE_DIM = SV.STATE_DIM
    C.NUM_ACTIONS = SV.NUM_ACTIONS
    C.HIDDEN = [256, 256]
    C.NUM_OUTCOMES = 3
    C.WIN_REWARD, C.LOSS_REWARD, C.DRAW_REWARD = 1.0, -1.0, -0.5
    C.WIN_TIME_DECAY = 0.0            # no urgency term: isolate the critic
    C.OUTCOME_REWARD = (C.WIN_REWARD, C.LOSS_REWARD, C.DRAW_REWARD)
    C.MAX_EPISODE_STEPS = args.t_limit
    C.ACTION_REPEAT = 1
    C.ROLLOUT_STEPS = args.rollout
    C.POS_SCALE = 1.0                 # obs[0:2] is already the agent's x,y
    C.AUX_OPP_ACTION_COEF = args.aux_action
    C.AUX_SELF_ACTION_COEF = args.aux_action
    C.AUX_NEXT_POS_COEF = args.aux_pos
    C.VALUE_COEF = args.value_coef
    C.CRITIC_LAYERS = args.critic_layers
    C.CRITIC_DROPOUT = 0.0
    C.EPOCHS, C.CRITIC_EPOCHS = args.epochs, args.critic_epochs
    C.MINIBATCH = 8192
    C.AUX_POS_HORIZONS = (1, 4, 15, 30)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--rollout", type=int, default=40_000)
    ap.add_argument("--t-limit", type=int, default=3000)
    ap.add_argument("--aux-action", type=float, default=0.10,
                    help="ppo4's default; 0 tests the 'aux heads are eating "
                         "the critic' hypothesis")
    ap.add_argument("--aux-pos", type=float, default=0.25)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--critic-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--critic-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C = configure(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from bonk4.pool import EpisodePool, actor_rows, critic_batches
    from bonk4.ppo import PPOAgent, entropy_coef_at

    E, D, NA = args.envs, SV.STATE_DIM, SV.NUM_ACTIONS
    venv = SV.VecSlimeVolley(E, seed=args.seed, t_limit=args.t_limit)
    agent = PPOAgent(D, NA, "cpu")
    rew = np.asarray(C.OUTCOME_REWARD, dtype=np.float32)

    # One ring + one pool PER SEAT: self-play, both driven by the same weights.
    ring = args.t_limit + 2
    R = {s: {k: np.zeros((ring, E) + shp, dtype=dt)
             for k, shp, dt in (("s", (D,), np.float32), ("a", (), np.int64),
                                ("lp", (), np.float32), ("v", (), np.float32))}
         for s in (0, 1)}
    pools = {s: EpisodePool(args.rollout + args.t_limit * 4 + 16, D) for s in (0, 1)}
    h = {s: agent.critic.initial_state(E, agent.device) for s in (0, 1)}
    w, ep_len = 0, np.zeros(E, dtype=np.int64)

    print(f"ppo5: Slime Volleyball, terminal-only 3-outcome | E={E} "
          f"rollout={args.rollout:,} t_limit={args.t_limit}")
    print(f"  actor {C.HIDDEN} shared encoder | critic GRUx{C.CRITIC_LAYERS}"
          f"({C.CRITIC_HIDDEN}) | aux_action={args.aux_action} "
          f"aux_pos={args.aux_pos} value_coef={args.value_coef}")
    print(f"  DEATH reference: ev=0.011 after 4M episodes\n")

    eps_done, t0 = 0, time.perf_counter()
    for upd in range(1, args.updates + 1):
        while pools[0].n < args.rollout:
            obs = {0: venv.o0.copy(), 1: venv.o1.copy()}
            act = {}
            for s in (0, 1):
                a, lp, v, _p, h[s] = agent.act_batch(obs[s], h[s])
                act[s] = a
                R[s]["s"][w] = obs[s]; R[s]["a"][w] = a
                R[s]["lp"][w] = lp; R[s]["v"][w] = v
            ep_len += 1
            done, out0, out1 = venv.step(act[0], act[1])
            if done.any():
                idxs = np.nonzero(done)[0]
                for i in idxs:
                    k = int(ep_len[i]); ep_len[i] = 0
                    sel = (np.arange(w - k + 1, w + 1) % ring)
                    for s, outc in ((0, out0), (1, out1)):
                        c = int(outc[i])
                        pools[s].add(R[s]["s"][sel, i], R[s]["a"][sel, i],
                                     R[s]["lp"][sel, i], R[s]["v"][sel, i],
                                     R[1 - s]["a"][sel, i], R[s]["a"][sel, i],
                                     c, float(rew[c]))
                    venv.reset_one(int(i))
                    eps_done += 1
                ti = torch.as_tensor(idxs)
                h[0][ti] = 0.0
                h[1][ti] = 0.0
            w = (w + 1) % ring

        batches = list(critic_batches(pools[0], C.CRITIC_LENGTH_BUCKETS, D))
        batches += list(critic_batches(pools[1], C.CRITIC_LENGTH_BUCKETS, D))
        stats = agent.update_critic_episodes(batches)
        parts = [actor_rows(pools[s]) for s in (0, 1)]
        stats.update(agent.update_actor(
            {"states": np.concatenate([p[0] for p in parts]),
             "actions": np.concatenate([p[1] for p in parts]),
             "logps": np.concatenate([p[2] for p in parts]),
             "advantages": np.concatenate([p[3] for p in parts])},
            entropy_coef_at(eps_done)))

        P = pools[0]
        lens = np.array([e[1] for e in P.eps])
        cls = np.array([e[2] for e in P.eps])
        Rrow = np.concatenate([np.full(k, rew[c]) for _o, k, c, _r in P.eps])
        A = Rrow - P.v[:P.n]
        ev = 1.0 - A.var() / max(Rrow.var(), 1e-9)
        print(f"  upd {upd:>3} | eps {eps_done:>6,} | len {lens.mean():5.0f} "
              f"| W/L/D {(cls==0).mean()*100:3.0f}/{(cls==1).mean()*100:3.0f}/"
              f"{(cls==2).mean()*100:3.0f} | ev {ev:+.3f} "
              f"oacc {stats['outcome_acc']*100:4.0f}% c {stats['critic_loss']:.3f} "
              f"H {stats['entropy']:.3f} | {time.perf_counter()-t0:5.0f}s")
        for s in (0, 1):
            pools[s].clear()


if __name__ == "__main__":
    main()
