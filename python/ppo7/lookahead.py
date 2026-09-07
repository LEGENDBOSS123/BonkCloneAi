"""N-rollout lookahead ("poor man's search") -- option #1 from the tree-search
survey: at each decision, sample K candidate (action, duration) pairs from the
actor, fork a real physics rollout for each one, and score it. No tree, no
node/edge/backup bookkeeping -- just K independent rollouts using
`LagEnv.snapshot_state()`/`restore_state()` to fork and rewind around each
candidate so the real episode is left exactly as it was.

ROLLOUT SHAPE (third iteration -- see history below). For each candidate:
  1. COMMIT for `hold_cycles` (capped at `depth_cap_cycles`): the searching
     seat is locked onto the candidate action, exactly like a real FiGAR hold.
  2. CONTINUE for `m_rollout_cycles` more decisions: BOTH seats now play
     NORMALLY -- sampled from the SAME actor, each honouring its own FiGAR
     hold via a real `HoldTracker`, exactly as `trainer.py`'s collection loop
     drives a seat. Not frozen, not a special-cased "always re-decide"
     opponent -- literally "how they would normally play."
  3. SCORE the resulting leaf with the critic, using the searching seat's
     hidden state as it actually stands after being properly stepped through
     the whole K+M window (the minGRU hidden always advances every cycle,
     independent of whether that cycle's action was held -- see
     `trainer.py`'s own note on this) -- not a stale pre-decision hidden.
  A terminal outcome anywhere in the window is scored EXACTLY (the true
  result), never approximated by the critic.

WHY THIS SHAPE, NOT THE EARLIER ONES. Two prior versions were tried and
measured (self-play vs. the plain policy from the SAME checkpoint):
  * One-ply, frozen opponent, leaf scored with a stale hidden: LOST 57-59%.
  * One-ply, opponent resampled every cycle (no hold), same stale-hidden leaf:
    no better (~59% loss) -- freezing wasn't the dominant problem.
  * One-ply, exhaustive over all 18 actions instead of K samples: LOST 90%.
    Enumerating actions the policy already considers unlikely exposes the
    critic to states it was never trained to be reliable on -- the classic
    value-overestimation failure mode (why offline-RL methods like CQL exist,
    and why real MCTS/PUCT weights search by the policy PRIOR instead of
    trusting raw value-argmax). This is the strongest evidence from the
    earlier rounds: candidates should stay close to the policy's own
    distribution, not exhaustively include what it already rejects.
  * Larger K (16) in sampled mode: best of the one-ply variants (still a
    loss), suggesting more genuine continuation, not more candidates alone,
    was the missing piece.
This version's M-step continuation directly targets what all four one-ply
attempts shared: scoring a leaf from a single fixed/near-fixed action with an
under-warmed hidden state. Playing both seats out normally for longer keeps
the trajectory on-distribution for the critic (both sides are doing what the
policy would actually do) and gives the hidden state real context to score
from, rather than asking the critic to extrapolate from one shallow ply.

CANDIDATE GENERATION (`candidate_mode`): "sample" draws `k` (action,
duration) pairs from the actor's own distribution (recommended, per the
finding above); "all" enumerates every action (measured much worse, kept only
for comparison).

PPO CORRECTNESS. `old_logp` for whichever candidate wins is recorded against
the RAW actor's probability of that (action, duration) pair, never a
"best-of-K" meta-distribution -- the same behavior/target split already used
by the exploiter's parameter-noise collection (`ppo.py`'s `behavior_actor`):
the thing that ACTS need not be the thing whose log-prob trains the PPO ratio,
as long as `old_logp` is honest about which distribution actually produced
the action.

COST. Plain step() ~10.6us, snapshot+restore ~295us (~28x a step -- see
`LagEnv.snapshot_state`'s docstring). Each candidate now simulates roughly
`depth_cap_cycles + m_rollout_cycles` decisions with TWO forward passes per
cycle (both seats), so cost scales with that sum, not just K -- measure before
choosing values for anything beyond evaluation-scale use. Not wired into the
multiprocess `VecCollector` training path; that is a separate, larger step
this module deliberately does not take.
"""

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .env import LagEnv
from .figar import HoldTracker


class LookaheadPolicy:
    """Single-seat, single-env policy: K-candidate rollout search instead of a
    plain sample. Drop-in for anywhere a `PPOAgent` + one `LagEnv` already
    drive a seat one decision at a time (play.py, a self-play eval loop) --
    NOT wired into the multiprocess `VecCollector` training path.

        pol = LookaheadPolicy(agent, env, seat=1, k=4)
        pol.reset_env()                    # new episode
        action, hold_cycles, logp = pol.act()
    """

    def __init__(self, agent, env: LagEnv, seat: int, k: int = 4,
                depth_cap_cycles: int = 8, m_rollout_cycles: int = 8,
                candidate_mode: str = "sample"):
        if not hasattr(env.sim, "get_full_state"):
            raise ValueError("lookahead needs the real engine (config.ENGINE "
                             "== 'real'): snapshot_state/restore_state go "
                             "through EngineSim.get_full_state/set_full_state")
        if candidate_mode not in ("sample", "all"):
            raise ValueError(f"candidate_mode must be 'sample' or 'all', "
                             f"got {candidate_mode!r}")
        self.agent = agent
        self.env = env
        self.seat = seat
        self.opp = 1 - seat
        self.k = max(1, int(k))
        self.candidate_mode = candidate_mode
        # Candidate commitment is capped (see module docstring); the
        # continuation is a separate, independent budget.
        self.depth_cap_cycles = depth_cap_cycles
        self.m_rollout_cycles = m_rollout_cycles
        self.recurrent = bool(getattr(C, "RECURRENT", False))
        self.h = torch.zeros(1, C.MINGRU_HIDDEN) if self.recurrent else None
        self._durs = np.asarray(C.DURATIONS, dtype=np.int64)
        self.hold = 0
        self.held_action = 0
        self.held_logp = 0.0

    def reset_env(self) -> None:
        self.hold = 0
        if self.recurrent:
            self.h = torch.zeros(1, C.MINGRU_HIDDEN)

    def act(self):
        """One decision-cycle. Returns (action_index, hold_cycles, logp) --
        `hold_cycles` and `logp` are only meaningful on a FREE decision (the
        caller is expected to hold the action itself otherwise, same contract
        as `HoldTracker`); `logp` is always the RAW actor's log-prob of the
        returned action (see PPO CORRECTNESS in the module docstring). Owns
        its own hold/hidden bookkeeping (`self.hold`/`self.h`); for a caller
        that owns its OWN per-env hidden-state array instead (e.g. sparse
        search-augmented training rows), use `search(hidden)` directly."""
        if self.hold > 0:
            self.hold -= 1
            return self.held_action, 0, self.held_logp
        a_idx, hold_cycles, logp, self.h = self.search(self.h)
        self.held_action, self.held_logp = a_idx, logp
        self.hold = hold_cycles - 1
        return a_idx, hold_cycles, logp

    def search(self, hidden):
        """The core K-candidate search from an EXTERNAL hidden state, on a
        decision known to be FREE (the caller decides that). Returns
        (action_index, hold_cycles, logp, hidden_after_one_real_step_at_that_
        action) -- the returned hidden is a single real step at the CHOSEN
        candidate, not the deep speculative hidden used only for scoring
        inside `_simulate_and_score`. Stateless w.r.t. `self.h`/`self.hold`,
        so an external caller (e.g. `Trainer`, which carries hidden state in
        its own per-env arrays) can call this directly without a full
        `LookaheadPolicy` owning that state."""
        state = self.env.decision_state(self.seat)
        hf = np.zeros(1, dtype=np.float32)   # free -> hold feature 0
        x = torch.from_numpy(np.concatenate([state, hf])).float().unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                logits, _ = self.agent.actor.step(x, hidden)
            else:
                logits = self.agent.actor(x)
            la, ld = logits[:, :C.NUM_ACTIONS], logits[:, C.NUM_ACTIONS:]
            lpa, lpd = F.log_softmax(la, 1), F.log_softmax(ld, 1)
            if self.candidate_mode == "all":
                # every action, each with its OWN independently-sampled
                # duration (the duration head doesn't depend on which action).
                a_cand = torch.arange(C.NUM_ACTIONS)
                d_cand = torch.multinomial(
                    lpd.exp().expand(C.NUM_ACTIONS, -1), 1).squeeze(1)
            else:
                a_cand = torch.multinomial(lpa.exp(), self.k, replacement=True)[0]
                d_cand = torch.multinomial(lpd.exp(), self.k, replacement=True)[0]
            cand_logp = lpa[0, a_cand] + lpd[0, d_cand]   # [n_candidates]

        snap = self.env.snapshot_state()
        best_i, best_score = 0, -1e18
        for i in range(len(a_cand)):
            a_idx, d_idx = int(a_cand[i]), int(d_cand[i])
            score = self._simulate_and_score(a_idx, int(self._durs[d_idx]))
            self.env.restore_state(snap)
            if score > best_score:
                best_i, best_score = i, score

        a_idx, d_idx = int(a_cand[best_i]), int(d_cand[best_i])
        logp = float(cand_logp[best_i])
        h_after = hidden
        if self.recurrent:
            with torch.no_grad():
                _, h_after = self.agent.actor.step(x, hidden)   # real step, chosen action
        hold_cycles = int(self._durs[d_idx])
        return a_idx, hold_cycles, logp, h_after

    def _act_seat_normally(self, seat: int, hold: HoldTracker, hidden):
        """One cycle of NORMAL play for `seat`: the net steps every cycle
        (hidden always advances -- FiGAR holds gate which ACTION applies, not
        whether the GRU advances, matching `trainer.py`'s collection loop
        exactly), and `hold` decides whether today's fresh sample is actually
        used or the seat's already-held action repeats. Reuses `HoldTracker`
        bit-for-bit rather than reimplementing hold bookkeeping by hand."""
        state = self.env.decision_state(seat)
        hf = hold.feature()                          # [1,1], BEFORE apply()
        x = torch.from_numpy(np.concatenate([state, hf[0]])).float().unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                logits, hidden = self.agent.actor.step(x, hidden)
            else:
                logits = self.agent.actor(x)
            pa = F.softmax(logits[:, :C.NUM_ACTIONS], 1).numpy()
            pd = F.softmax(logits[:, C.NUM_ACTIONS:], 1).numpy()
            a = np.array([np.random.choice(C.NUM_ACTIONS, p=pa[0])])
            d = np.array([np.random.choice(C.NUM_DURATIONS, p=pd[0])])
        applied, _ = hold.apply(a, d)
        return int(applied[0]), hidden

    def _simulate_and_score(self, a_idx: int, hold_cycles: int) -> float:
        """Fork: commit to `a_idx` for `hold_cycles` (capped), then let both
        seats play `m_rollout_cycles` more decisions NORMALLY, then score the
        leaf. See the module docstring's ROLLOUT SHAPE section."""
        cycles = min(hold_cycles, self.depth_cap_cycles)
        my_hold = HoldTracker(1)
        my_hold.hold[0], my_hold.held[0] = cycles, a_idx   # locked candidate
        opp_hold = HoldTracker(1)                          # free from the start
        my_h = self.h.clone() if self.recurrent else None
        opp_h = torch.zeros(1, C.MINGRU_HIDDEN) if self.recurrent else None

        for _ in range(cycles + self.m_rollout_cycles):
            my_action, my_h = self._act_seat_normally(self.seat, my_hold, my_h)
            opp_action, opp_h = self._act_seat_normally(self.opp, opp_hold, opp_h)
            self.env.set_decision(self.seat, my_action)
            self.env.set_decision(self.opp, opp_action)
            for _ in range(C.ACTION_REPEAT):
                res = self.env.tick()
                if res["done"]:
                    dead_self, dead_opp = res["dead"][self.seat], res["dead"][self.opp]
                    if res["timeout"] or (dead_self and dead_opp):
                        return 0.0               # draw, on the atom scale
                    return -1.0 if dead_self else 1.0
        return self._score_leaf(my_h)

    def _score_leaf(self, hidden) -> float:
        leaf = self.env.decision_state(self.seat)
        hf = np.zeros(1, dtype=np.float32)
        x = torch.from_numpy(np.concatenate([leaf, hf])).float().unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                out, _ = self.agent.critic.step(x, hidden)
            else:
                out = self.agent.critic(x)
            if self.agent.cat:
                return float((torch.softmax(out, -1) @ self.agent.atoms)[0])
            return float(out.squeeze())
