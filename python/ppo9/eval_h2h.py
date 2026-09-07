"""Head-to-head evaluation — THE progress metric.

ELO here is a treadmill: it is measured against a rolling buffer of the main's
own recent selves, so it plateaus whether or not the agent is actually
improving. To know, play two checkpoints directly::

    python -m ppo9.eval_h2h --a runs/ppo9/late.json --b runs/ppo9/early.json

Sides are swapped for the second half so spawn asymmetry cancels, and lag is
randomized per episode exactly as in training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from bonk3.simadapter import make_sim

from bonkenv import EnvConfig, LagEnv, Outcome, build_layout
from .config import FigarConfig
from .policy import load_checkpoint, load_policy, resolve_agent


def play(w0: dict, w1: dict, cfg: EnvConfig, figar: FigarConfig,
         n_games: int, n_envs: int, greedy: bool) -> tuple[int, int, int]:
    """Seat 0 vs seat 1. Returns ``(seat0_wins, seat1_wins, draws)``.

    Policies are rebuilt here so each side gets its own hidden state sized to
    `n_envs`, and each env's memory is cleared when its episode ends.
    """
    layout = build_layout(cfg.obs)
    pol0, pol1 = load_policy(w0, figar), load_policy(w1, figar)
    pol0.reset(n_envs)
    pol1.reset(n_envs)
    envs = [LagEnv(make_sim(cfg.engine.map_name, engine=cfg.engine.engine),
                   cfg, layout) for _ in range(n_envs)]
    for e in envs:
        e.reset()

    wins0 = wins1 = draws = finished = 0
    tick = 0
    while finished < n_games:
        if tick % cfg.episode.action_repeat == 0:
            a0 = pol0.act(np.stack([e.decision_state(0) for e in envs]), greedy)
            a1 = pol1.act(np.stack([e.decision_state(1) for e in envs]), greedy)
            for i, e in enumerate(envs):
                e.set_decision(0, int(a0[i]))
                e.set_decision(1, int(a1[i]))
        for i, e in enumerate(envs):
            res = e.tick()
            if res.done:
                if res.outcome[0] == Outcome.WIN:
                    wins0 += 1
                elif res.outcome[0] == Outcome.LOSS:
                    wins1 += 1
                else:
                    draws += 1
                finished += 1
                e.reset()
                pol0.reset_env(i)
                pol1.reset_env(i)
        tick += 1
    return wins0, wins1, draws


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint glob for agent A")
    ap.add_argument("--b", required=True, help="checkpoint glob for agent B")
    ap.add_argument("--a-agent", default="main", help="main | snap:N | exp:N")
    ap.add_argument("--b-agent", default="main")
    ap.add_argument("--games", type=int, default=2000, help="games PER SIDE")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax actions (default: sample, as in training)")
    args = ap.parse_args()

    obj_a, path_a = load_checkpoint(args.a)
    obj_b, path_b = load_checkpoint(args.b)
    net_a, label_a = resolve_agent(obj_a, args.a_agent)
    net_b, label_b = resolve_agent(obj_b, args.b_agent)

    # Both sides must share an environment, so evaluation uses the default env
    # config. A checkpoint carries the config it trained under; if the two
    # disagree with each other the comparison is meaningless anyway.
    cfg = EnvConfig().resolve()
    figar = FigarConfig()

    print(f"A = {Path(path_a).name}  [{label_a}]")
    print(f"B = {Path(path_b).name}  [{label_b}]")
    print(f"engine {cfg.engine.engine} map {cfg.engine.map_name}")
    print(f"{args.games} games/side, {'greedy' if args.greedy else 'sampled'}, "
          f"lag {'randomized' if cfg.lag.randomize else 'fixed'}")

    # Swap sides for the second half so spawn asymmetry cancels.
    a1, b1, d1 = play(net_a, net_b, cfg, figar, args.games, args.envs, args.greedy)
    b2, a2, d2 = play(net_b, net_a, cfg, figar, args.games, args.envs, args.greedy)

    a_wins, b_wins, draws = a1 + a2, b1 + b2, d1 + d2
    total = a_wins + b_wins + draws
    score = (a_wins + 0.5 * draws) / total

    print(f"\n  A on seat0: A {a1} / B {b1} / draw {d1}")
    print(f"  A on seat1: A {a2} / B {b2} / draw {d2}")
    print(f"\n  A wins {a_wins} | B wins {b_wins} | draws {draws}")
    print(f"  A score (draws count half): {score * 100:.1f}%"
          f"   raw wins: {a_wins / total * 100:.1f}%")
    verdict = ("A strongly better — a real improvement" if score > 0.65 else
               "A slightly better" if score > 0.55 else
               "~even — no meaningful improvement over this span" if score > 0.45
               else "B better — A regressed")
    print(f"  verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
