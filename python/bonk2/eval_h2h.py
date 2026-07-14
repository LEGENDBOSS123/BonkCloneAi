"""bonk2 head-to-head evaluation — the FIXED-reference test that ELO can't give
you. ELO is measured against a rolling buffer of the main's recent selves, so
it plateaus whether or not the agent is improving. To actually know, play two
checkpoints directly:

    python -m bonk2.eval_h2h --a runs/ppo2/bonk2-ppo-ep6000000-*.json \
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
import torch

from bonk.networks import MLP
from bonk.sim import BonkSim

from . import config as C
from .env import LagEnv

REPO = Path(__file__).resolve().parents[2]


def load_actor(path_glob: str):
    matches = sorted(glob.glob(path_glob))
    if not matches:
        raise FileNotFoundError(path_glob)
    path = matches[-1]
    with open(path) as f:
        obj = json.load(f)
    if obj.get("stateDim") not in (None, C.STATE_DIM):
        raise ValueError(f"{path}: stateDim {obj.get('stateDim')} != "
                         f"{C.STATE_DIM} (v1 checkpoint?)")
    net = MLP.from_records(obj["agent"]["actor"])
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net, path


@torch.no_grad()
def act(net: MLP, states: np.ndarray, greedy: bool) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(states)).float()
    logits = net(x)
    if greedy:
        return logits.argmax(1).numpy()
    return torch.multinomial(torch.softmax(logits, 1), 1).squeeze(1).numpy()


def play(net0: MLP, net1: MLP, map_json: dict, n_games: int, n_envs: int,
         greedy: bool):
    """net0 on seat 0, net1 on seat 1. Returns (net0_wins, net1_wins, draws)."""
    envs = [LagEnv(BonkSim(map_json)) for _ in range(n_envs)]
    for e in envs:
        e.reset()
    w0 = w1 = draw = done_games = 0
    tick = 0
    while done_games < n_games:
        if tick % C.ACTION_REPEAT == 0:
            s0 = np.stack([e.decision_state(0) for e in envs])
            s1 = np.stack([e.decision_state(1) for e in envs])
            a0 = act(net0, s0, greedy)
            a1 = act(net1, s1, greedy)
            for i, e in enumerate(envs):
                e.set_decision(0, int(a0[i]))
                e.set_decision(1, int(a1[i]))
        for e in envs:
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
        tick += 1
    return w0, w1, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint glob for agent A")
    ap.add_argument("--b", required=True, help="checkpoint glob for agent B")
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"))
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax actions (default: sample, as in training)")
    args = ap.parse_args()

    net_a, path_a = load_actor(args.a)
    net_b, path_b = load_actor(args.b)
    with open(args.map) as f:
        map_json = json.load(f)
    print(f"A = {Path(path_a).name}")
    print(f"B = {Path(path_b).name}")
    print(f"{args.games} games/side, {'greedy' if args.greedy else 'sampled'}, "
          f"lag {'randomized' if C.INPUT_LAG_RANDOM else C.INPUT_LAG}")

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
