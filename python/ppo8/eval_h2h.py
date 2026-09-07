"""bonk4 head-to-head evaluation — the FIXED-reference test that ELO can't give
you. ELO is measured against a rolling buffer of the main's recent selves, so
it plateaus whether or not the agent is improving. To actually know, play two
checkpoints directly:

    python -m ppo6.eval_h2h --a runs/ppo2/bonk2-ppo-ep6000000-*.json \
                             --b runs/ppo2/bonk2-ppo-ep5000000-*.json --games 2000

~90% = real improvement over that span; ~50% = genuinely stuck. Sides are
swapped each half so spawn asymmetry cancels; lag is randomized per episode
exactly as in training.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from bonk3.simadapter import make_sim

from . import config as C
from .env import LagEnv
from .policy import load_policy, resolve_agent

REPO = Path(__file__).resolve().parents[2]


def load_actor(path_glob: str):
    """Returns (weights, path). Architecture is detected at build time by
    load_policy, so recurrent and feedforward checkpoints both work."""
    matches = sorted(glob.glob(path_glob))
    if not matches:
        raise FileNotFoundError(path_glob)
    path = matches[-1]
    with open(path) as f:
        obj = json.load(f)
    # ppo8 saves AGENT_STATE_DIM (obs + hold feature); accept either.
    if obj.get("stateDim") not in (None, C.STATE_DIM, C.AGENT_STATE_DIM):
        raise ValueError(f"{path}: stateDim {obj.get('stateDim')} != "
                         f"{C.AGENT_STATE_DIM}")
    weights, _ = resolve_agent(obj, "main")
    return weights, path


def play(w0_weights, w1_weights, map_json: dict, n_games: int, n_envs: int,
         greedy: bool):
    """seat 0 vs seat 1. Returns (seat0_wins, seat1_wins, draws).

    Policies are rebuilt here so each side gets its own hidden state, sized to
    n_envs. A recurrent policy MUST have that state cleared per env when its
    episode ends, or memory bleeds across games."""
    pol0, pol1 = load_policy(w0_weights), load_policy(w1_weights)
    pol0.reset(n_envs)
    pol1.reset(n_envs)
    src = C.MAP_NAME if C.ENGINE == "real" else map_json
    envs = [LagEnv(make_sim(src, engine=C.ENGINE)) for _ in range(n_envs)]
    for e in envs:
        e.reset()
    w0 = w1 = draw = done_games = 0
    tick = 0
    while done_games < n_games:
        if tick % C.ACTION_REPEAT == 0:
            s0 = np.stack([e.decision_state(0) for e in envs])
            s1 = np.stack([e.decision_state(1) for e in envs])
            a0 = pol0.act(s0, greedy)
            a1 = pol1.act(s1, greedy)
            for i, e in enumerate(envs):
                e.set_decision(0, int(a0[i]))
                e.set_decision(1, int(a1[i]))
        for i, e in enumerate(envs):
            res = e.tick()
            if res["done"]:
                dead0, dead1 = res["dead"]
                if res["timeout"] or (dead0 and dead1):
                    draw += 1
                elif dead1:
                    w0 += 1
                else:
                    w1 += 1
                done_games += 1
                e.reset()
                pol0.reset_env(i)
                pol1.reset_env(i)
        tick += 1
    return w0, w1, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint glob for agent A")
    ap.add_argument("--b", required=True, help="checkpoint glob for agent B")
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"),
                    help="legacy-engine map file; ignored when config.ENGINE "
                         "== 'real' (that uses config.MAP_NAME)")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax actions (default: sample, as in training)")
    args = ap.parse_args()

    net_a, path_a = load_actor(args.a)
    net_b, path_b = load_actor(args.b)
    # The real engine loads its own decoded map by name; only the legacy
    # backend needs this file, so don't fail on a missing one.
    map_json = None
    if C.ENGINE != "real":
        with open(args.map) as f:
            map_json = json.load(f)
    print(f"A = {Path(path_a).name}")
    print(f"B = {Path(path_b).name}")
    print(f"engine {C.ENGINE}"
          + (f" map {C.MAP_NAME}" if C.ENGINE == "real" else ""))
    print(f"{args.games} games/side, {'greedy' if args.greedy else 'sampled'}, "
          f"lag {'randomized' if C.LAG_RANDOM else f'self {C.SELF_LAG} view {C.VIEW_LAG}'}")

    # Side 1: A on seat 0. Side 2: A on seat 1 (swap to cancel spawn bias).
    a_w1, b_w1, d1 = play(net_a, net_b, map_json, args.games, args.envs,
                          args.greedy)
    b_w2, a_w2, d2 = play(net_b, net_a, map_json, args.games, args.envs,
                          args.greedy)

    a_wins, b_wins, draws = a_w1 + a_w2, b_w1 + b_w2, d1 + d2
    total = a_wins + b_wins + draws
    a_score = (a_wins + 0.5 * draws) / total   # draws split
    print(f"\n  A seat0: A {a_w1} / B {b_w1} / draw {d1}")
    print(f"  A seat1: A {a_w2} / B {b_w2} / draw {d2}")
    print(f"\n  A wins {a_wins} | B wins {b_wins} | draws {draws}")
    print(f"  A win rate (draws=½): {a_score*100:.1f}%   "
          f"raw: {a_wins/total*100:.1f}%")
    verdict = ("A strongly better — real improvement (flat ELO was a treadmill)"
               if a_score > 0.65 else
               "A slightly better" if a_score > 0.55 else
               "~even — no meaningful improvement over this span" if a_score > 0.45
               else "B better — A regressed")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
