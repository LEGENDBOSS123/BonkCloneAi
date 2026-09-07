"""Does N-rollout lookahead search actually help? Self-plays the SAME
checkpoint against itself, one seat driven by `LookaheadPolicy` (K-candidate
search) and the other by the plain policy (`policy.load_policy`, exactly what
training/eval/play use everywhere else) -- so any win-rate edge is
attributable to the search alone, not to a better or different network. Sides
are swapped each half so spawn asymmetry cancels, same convention as
`eval_h2h.py`.

    python -m ppo7.eval_lookahead --ckpt <path> --games 200 --k 4

This is a single-process, sequential-env loop (not the multiprocess
VecCollector), and each lookahead decision costs roughly K forked rollouts on
top of the plain policy's one -- expect it to run noticeably slower than
eval_h2h for the same game count. Keep --envs modest.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bonk3.simadapter import make_sim

from . import config as C
from .env import LagEnv
from .lookahead import LookaheadPolicy
from .policy import load_policy, resolve_agent
from .ppo import PPOAgent

REPO = Path(__file__).resolve().parents[2]


def _load_agent(ckpt_path):
    """PPOAgent rebuilt from a checkpoint's main actor+critic -- LookaheadPolicy
    needs the actual agent (for `.actor`/`.critic`/`.atoms`), not just the
    `policy.ActorPolicy` wrapper the plain side uses."""
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    agent = PPOAgent(C.AGENT_STATE_DIM, C.NUM_ACTIONS, device="cpu")
    agent.actor.load_records(ckpt["agent"]["actor"])
    agent.critic.load_records(ckpt["agent"]["critic"])
    return agent, ckpt


def play(agent, ckpt, search_seat: int, n_games: int, n_envs: int, k: int,
        candidate_mode: str, depth_cap_cycles: int, m_rollout_cycles: int):
    """search_seat plays via LookaheadPolicy; 1-search_seat plays via the
    plain policy -- BOTH loaded from the SAME checkpoint (`agent` and `ckpt`
    come from one `_load_agent` call), so any edge is the search alone.
    Returns (search_wins, plain_wins, draws)."""
    plain_seat = 1 - search_seat
    plain_recs, _ = resolve_agent(ckpt, "main")
    plain = load_policy(plain_recs)
    plain.reset(n_envs)

    envs = [LagEnv(make_sim(C.MAP_NAME, engine=C.ENGINE)) for _ in range(n_envs)]
    for e in envs:
        e.reset()
    searchers = [LookaheadPolicy(agent, e, seat=search_seat, k=k,
                                 candidate_mode=candidate_mode,
                                 depth_cap_cycles=depth_cap_cycles,
                                 m_rollout_cycles=m_rollout_cycles)
                for e in envs]
    for s in searchers:
        s.reset_env()

    w_search = w_plain = draw = done_games = 0
    tick = 0
    while done_games < n_games:
        if tick % C.ACTION_REPEAT == 0:
            s_plain = np.stack([e.decision_state(plain_seat) for e in envs])
            a_plain = plain.act(s_plain, greedy=False)
            for i, e in enumerate(envs):
                a_search, _, _ = searchers[i].act()
                e.set_decision(search_seat, a_search)
                e.set_decision(plain_seat, int(a_plain[i]))
        for i, e in enumerate(envs):
            res = e.tick()
            if res["done"]:
                dead_s, dead_p = res["dead"][search_seat], res["dead"][plain_seat]
                if res["timeout"] or (dead_s and dead_p):
                    draw += 1
                elif dead_p:
                    w_search += 1
                else:
                    w_plain += 1
                done_games += 1
                e.reset()
                plain.reset_env(i)
                searchers[i].reset_env()
        tick += 1
    return w_search, w_plain, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=100,
                    help="games PER SIDE (total = 2x this)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--k", type=int, default=4,
                    help="candidates/decision in 'sample' mode (ignored by 'all')")
    ap.add_argument("--candidates", choices=("sample", "all"), default="sample",
                    help="'sample'=K stochastic draws, 'all'=every action, each "
                    "with its own sampled duration (measured much worse -- kept "
                    "for comparison only)")
    ap.add_argument("--depth-cap", type=int, default=8,
                    help="cycles the candidate action is committed to (capped)")
    ap.add_argument("--m-rollout", type=int, default=8,
                    help="additional cycles BOTH seats then play normally, "
                    "before the leaf is scored")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent, ckpt = _load_agent(args.ckpt)
    print(f"checkpoint: {Path(args.ckpt).name}")
    print(f"lookahead candidates={args.candidates}"
         f"{f' K={args.k}' if args.candidates == 'sample' else ''}, "
         f"depth_cap={args.depth_cap} + m_rollout={args.m_rollout} cycles, "
         f"{args.games} games/side, {args.envs} envs")

    kw = dict(k=args.k, candidate_mode=args.candidates,
             depth_cap_cycles=args.depth_cap, m_rollout_cycles=args.m_rollout)
    s1, p1, d1 = play(agent, ckpt, search_seat=0, n_games=args.games,
                      n_envs=args.envs, **kw)
    s2, p2, d2 = play(agent, ckpt, search_seat=1, n_games=args.games,
                      n_envs=args.envs, **kw)

    s_wins, p_wins, draws = s1 + s2, p1 + p2, d1 + d2
    total = s_wins + p_wins + draws
    s_score = (s_wins + 0.5 * draws) / total
    print(f"\n  search seat0: search {s1} / plain {p1} / draw {d1}")
    print(f"  search seat1: search {s2} / plain {p2} / draw {d2}")
    print(f"\n  search wins {s_wins} | plain wins {p_wins} | draws {draws}")
    print(f"  search win rate (draws=½): {s_score*100:.1f}%   "
         f"raw: {s_wins/total*100:.1f}%")
    verdict = ("search meaningfully helps" if s_score > 0.58 else
               "search slightly helps" if s_score > 0.52 else
               "~even -- search isn't doing anything measurable" if s_score > 0.48
               else "search HURTS (worse than the plain policy)")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
