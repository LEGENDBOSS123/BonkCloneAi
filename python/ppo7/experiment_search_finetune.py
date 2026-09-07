"""Does sparse search-augmented collection actually improve training, over a
real (if small-scale) PPO run? For a small PERCENTAGE of the learner's FREE
decisions, `LookaheadPolicy.search()` picks the action instead of a plain
sample -- literally "use search's action as that row's real collected
action": `old_logp` stays the RAW actor's log-prob of whichever action
actually gets used, so PPO's clipped-surrogate/ratio machinery needs no
changes at all, and every non-searched row (the vast majority) collects at
full normal speed.

    python -m ppo7.experiment_search_finetune --ckpt <path> --updates 6 \
        --search-frac 0.03 --envs 24

Runs TWO trainers from the SAME starting weights for the SAME number of
updates -- one plain (control), one with sparse search-augmented rows
(treatment) -- then head-to-head evaluates the two resulting checkpoints
against each other (both against the SAME unmodified starting weights too, as
a sanity floor) so any edge is attributable to the sparse search rows alone.

Single-process (`SingleProcessCollector`), pure self-play, no league --
scoped as a PROTOTYPE to see if the idea has legs before committing to wiring
it into the real multiprocess `VecCollector` training path, which is a much
bigger, separate step (workers would need actor+critic weights shipped to
them, which the real trainer deliberately never does today).
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import eval_lookahead
from .lookahead import LookaheadPolicy
from .ppo import PPOAgent
from .single_process_collector import SingleProcessCollector
from .trainer import LEARNER, OPPONENT, Trainer

REPO = Path(__file__).resolve().parents[2]


class SearchAugmentedTrainer(Trainer):
    """`Trainer` where a small FRACTION of the learner's FREE decisions each
    cycle are picked by `LookaheadPolicy.search()` instead of a plain sample.
    Needs `SingleProcessCollector` specifically: search must reach the live
    `LagEnv` objects to fork/restore around candidates, which the real
    multiprocess `VecCollector` never exposes to the trainer process.
    """

    def __init__(self, collector: SingleProcessCollector, agent: PPOAgent,
                search_frac: float = 0.02, k: int = 4,
                depth_cap_cycles: int = 8, m_rollout_cycles: int = 8,
                seed: int = 0):
        super().__init__(collector, agent)
        self.search_frac = search_frac
        self.k, self.depth_cap_cycles, self.m_rollout_cycles = (
            k, depth_cap_cycles, m_rollout_cycles)
        self._rng = np.random.default_rng(seed)
        self.searched_rows = 0
        self.free_rows = 0

    def _infer_recurrent(self, obs):
        if not self.recurrent:
            raise ValueError("SearchAugmentedTrainer requires config.RECURRENT")
        active = self._active_agent()
        E = self.E
        actions = np.zeros((E, 2), dtype=np.int64)

        l_states = self._seat_states(obs, LEARNER)
        self.buf.set_hidden(self.h_a, self.h_c)
        sa, sd, slp, sv, h_a_new, h_c_new = active.act_step(
            l_states, self.h_a, self.h_c)

        # Which rows are genuinely FREE this cycle (a real decision, not a
        # forced repeat) -- only those are eligible for search, matching what
        # `HoldTracker.apply` itself would treat as free.
        free0 = self.hold[LEARNER].free_mask()
        self.free_rows += int(free0.sum())
        pick = free0 & (self._rng.random(E) < self.search_frac)
        for i in np.nonzero(pick)[0]:
            env = self.coll.envs[i]
            lp = LookaheadPolicy(active, env, seat=LEARNER, k=self.k,
                                 depth_cap_cycles=self.depth_cap_cycles,
                                 m_rollout_cycles=self.m_rollout_cycles)
            h_before = torch.from_numpy(self.h_a[i:i + 1])
            a_idx, hold_cycles, logp, h_after = lp.search(h_before)
            sa[i] = a_idx
            sd[i] = int(np.searchsorted(self._durs_np, hold_cycles))
            slp[i] = logp
            h_a_new[i] = h_after.numpy()[0]
            self.searched_rows += 1
        self.h_a, self.h_c = h_a_new, h_c_new

        applied, free = self.hold[LEARNER].apply(sa, sd)
        actions[:, LEARNER] = applied
        self.buf.set_learner(l_states, applied, sd, free, slp, sv)
        self.buf.open_opponent()        # seat 1 is not trained in this mode

        s1 = self._seat_states(obs, OPPONENT)
        for key, idxs in self._opponent_groups().items():
            kind = key[0]
            if kind == "idle":
                actions[idxs, OPPONENT] = 0
                continue
            net = active.actor if kind == "current" else self.league.opponent_net(*key)
            a_op, d_op = self._act_opponent_group(active, net, s1, idxs)
            applied_o, _ = self.hold[OPPONENT].apply(a_op, d_op, idxs)
            actions[idxs, OPPONENT] = applied_o
        return actions

    @property
    def _durs_np(self):
        return np.asarray(C.DURATIONS, dtype=np.int64)


def run_updates(trainer: Trainer, n_updates: int, log_prefix: str):
    updates_done = 0
    while updates_done < n_updates:
        trainer.cycle()
        if trainer.learner_steps() >= C.ROLLOUT_STEPS:
            stats, _ = trainer.run_update()
            updates_done += 1
            extra = ""
            if isinstance(trainer, SearchAugmentedTrainer):
                extra = f" searched={trainer.searched_rows}/{trainer.free_rows} free rows"
            print(f"  {log_prefix} update {updates_done}/{n_updates}: "
                 f"a={stats['actor_loss']:.4f} c={stats['critic_loss']:.4f} "
                 f"H={stats['entropy']:.3f} hold={stats['mean_hold']:.1f}{extra}")
    return trainer


def _fake_ckpt(agent: PPOAgent) -> dict:
    """The minimal checkpoint shape `eval_lookahead.play`/`resolve_agent` need
    to build a PLAIN-policy opponent, built from an in-memory agent's own
    current records -- lets the exploitability probe below reuse
    `eval_lookahead.play` exactly as written, no disk round-trip."""
    return {"agent": {"actor": agent.actor.to_records()}}


def exploitability(agent: PPOAgent, n_games: int, n_envs: int, k: int,
                   depth_cap_cycles: int, seed: int) -> float:
    """How exploitable is `agent`'s CRITIC, standardized probe: exhaustive-
    action search (`candidate_mode="all"`, `m_rollout_cycles=0` -- the exact
    configuration that broke an untrained checkpoint down to ~10% by finding
    actions the critic overestimates) against the agent's own plain policy,
    same checkpoint both sides. Returns the searcher's win rate (draws=1/2);
    LOWER is MORE exploitable (a well-calibrated critic should make this probe
    struggle to find an edge, i.e. push this back toward 50%)."""
    ckpt = _fake_ckpt(agent)
    kw = dict(k=k, candidate_mode="all", depth_cap_cycles=depth_cap_cycles,
             m_rollout_cycles=0)
    torch.manual_seed(seed); np.random.seed(seed)
    s1, p1, d1 = eval_lookahead.play(agent, ckpt, search_seat=0,
                                     n_games=n_games, n_envs=n_envs, **kw)
    s2, p2, d2 = eval_lookahead.play(agent, ckpt, search_seat=1,
                                     n_games=n_games, n_envs=n_envs, **kw)
    s, p, d = s1 + s2, p1 + p2, d1 + d2
    total = s + p + d
    return (s + 0.5 * d) / total if total else 0.5


def h2h(agent_a: PPOAgent, agent_b: PPOAgent, n_games: int, n_envs: int, seed: int):
    """Self-play A vs B, sides swapped each half. Reuses SingleProcessCollector's
    env-construction convenience via a fresh pair of collectors' worth of envs."""
    from bonk3.simadapter import make_sim

    from .env import LagEnv
    from .figar import HoldTracker

    def play(seat_a):
        seat_b = 1 - seat_a
        envs = [LagEnv(make_sim(C.MAP_NAME, engine=C.ENGINE)) for _ in range(n_envs)]
        for e in envs:
            e.reset()
        h = {0: torch.zeros(n_envs, C.MINGRU_HIDDEN), 1: torch.zeros(n_envs, C.MINGRU_HIDDEN)}
        hold = {0: HoldTracker(n_envs), 1: HoldTracker(n_envs)}
        agents = {seat_a: agent_a, seat_b: agent_b}
        wa = wb = draw = done_games = 0
        tick = 0
        while done_games < n_games:
            if tick % C.ACTION_REPEAT == 0:
                acts = {}
                for seat in (0, 1):
                    s = np.stack([e.decision_state(seat) for e in envs])
                    hf = hold[seat].feature()
                    x = torch.from_numpy(np.concatenate([s, hf], 1)).float()
                    with torch.no_grad():
                        logits, h[seat] = agents[seat].actor.step(x, h[seat])
                        pa = torch.softmax(logits[:, :C.NUM_ACTIONS], 1)
                        pd = torch.softmax(logits[:, C.NUM_ACTIONS:], 1)
                        a = torch.multinomial(pa, 1).squeeze(1).numpy()
                        d = torch.multinomial(pd, 1).squeeze(1).numpy()
                    acts[seat], _ = hold[seat].apply(a, d)
                for i, e in enumerate(envs):
                    e.set_decision(0, int(acts[0][i]))
                    e.set_decision(1, int(acts[1][i]))
            for i, e in enumerate(envs):
                res = e.tick()
                if res["done"]:
                    dead_a, dead_b = res["dead"][seat_a], res["dead"][seat_b]
                    if res["timeout"] or (dead_a and dead_b):
                        draw += 1
                    elif dead_b:
                        wa += 1
                    else:
                        wb += 1
                    done_games += 1
                    e.reset()
                    for seat in (0, 1):
                        hold[seat].hold[i] = 0
                        h[seat][i] = 0.0
            tick += 1
        return wa, wb, draw

    torch.manual_seed(seed); np.random.seed(seed)
    a1, b1, d1 = play(seat_a=0)
    a2, b2, d2 = play(seat_a=1)
    a_wins, b_wins, draws = a1 + a2, b1 + b2, d1 + d2
    total = a_wins + b_wins + draws
    return (a_wins + 0.5 * draws) / total, a_wins, b_wins, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="starting checkpoint (every "
                    "arm warm-starts the actor+critic from this)")
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--rollout-steps", type=int, default=2048)
    ap.add_argument("--search-frac", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth-cap", type=int, default=8)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--eval-envs", type=int, default=12)
    ap.add_argument("--exploit-games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C.LEAGUE_ENABLED = False       # pure self-play; keep this a clean A/B on the idea

    with open(args.ckpt) as f:
        start_ckpt = json.load(f)
    start_actor = start_ckpt["agent"]["actor"]
    start_critic = start_ckpt["agent"]["critic"]

    def fresh_agent():
        a = PPOAgent(C.AGENT_STATE_DIM, C.NUM_ACTIONS, device="cpu")
        a.actor.load_records(copy.deepcopy(start_actor))
        a.critic.load_records(copy.deepcopy(start_critic))
        return a

    # arm name -> m_rollout_cycles. "m8" = commit + realistic M-step
    # continuation (the config that reached parity earlier); "m0" = commit
    # only, score DIRECTLY with no continuation -- the "ditch the multilevel
    # thing" simplification, made possible because `_act_seat_normally`
    # already lets the OPPONENT play normally (own hold, own policy) even
    # during the commit phase, so this isn't the old frozen-opponent one-ply.
    arms = {"control": None, "treat_m8": 8, "treat_m0": 0}

    print(f"starting weights: {Path(args.ckpt).name}")
    print(f"{args.updates} updates/arm, rollout_steps={args.rollout_steps}, "
         f"{args.envs} envs, search_frac={args.search_frac}, k={args.k}, "
         f"depth_cap={args.depth_cap}\n")

    agents = {"start": fresh_agent()}
    for name, m_rollout in arms.items():
        C.ROLLOUT_STEPS = args.rollout_steps
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        agent = fresh_agent()
        if m_rollout is None:
            trainer = Trainer(SingleProcessCollector(args.envs), agent)
        else:
            trainer = SearchAugmentedTrainer(
                SingleProcessCollector(args.envs), agent,
                search_frac=args.search_frac, k=args.k,
                depth_cap_cycles=args.depth_cap, m_rollout_cycles=m_rollout,
                seed=args.seed)
        t0 = time.perf_counter()
        run_updates(trainer, args.updates, name)
        extra = ""
        if isinstance(trainer, SearchAugmentedTrainer):
            extra = (f" (searched {trainer.searched_rows}/{trainer.free_rows}"
                     f" = {trainer.searched_rows/max(1,trainer.free_rows)*100:.2f}%)")
        print(f"  {name} done in {time.perf_counter()-t0:.1f}s{extra}\n")
        agents[name] = agent

    print("head-to-head (row vs column, row's win rate):")
    names = list(agents)
    for a in names:
        row = []
        for b in names:
            if a == b:
                row.append("  -- ")
                continue
            score, *_ = h2h(agents[a], agents[b], args.eval_games,
                            args.eval_envs, args.seed)
            row.append(f"{score*100:5.1f}")
        print(f"  {a:10s} " + " ".join(row))
    print("             " + " ".join(f"{n:>5s}" for n in names))

    print("\ncritic exploitability (exhaustive-action search vs own plain "
         "policy; LOWER = the critic is MORE exploitable):")
    for name, agent in agents.items():
        score = exploitability(agent, args.exploit_games, args.eval_envs,
                               args.k, args.depth_cap, args.seed)
        print(f"  {name:10s} {score*100:5.1f}%")


if __name__ == "__main__":
    main()
