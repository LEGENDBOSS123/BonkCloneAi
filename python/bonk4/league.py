"""bonk2 league: opponent pool + adaptive exploiter generation (AlphaStar-lite).

One module owns everything population-related, so the trainer stays a pure
PPO/collection loop:

- snapshots: frozen past selves, taken every SNAPSHOT_INTERVAL main episodes.
- exploiters: graduated best-responses to a frozen main. A new exploiter phase
  starts when the main has MASTERED its pool (no frozen agent still beats it by
  more than EXPLOITER_TRIGGER_WR), with min-spacing and a max-interval
  diversity floor. During a phase, main training pauses and exp_agent trains
  against frozen_main; the phase ends on a winrate gate + sharpen tail, or the
  timeout.
- opponent sampling (main phase): 30% mirror match vs the live current model,
  60% PFSP over the whole pool (weighted by the main's live loss rate — whoever
  the main loses to most gets the most games), 10% uniform over the pool (keeps
  loss rates fresh vs "solved" opponents). Empty pool -> current.

Eviction from a full pool returns True so the caller can shift any in-flight
opponent indices.
"""

import numpy as np

from bonk.networks import MLP

from . import config as C
from .ppo import PPOAgent


def elo_update(ra, rb, score_a, k=C.ELO_K):
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


class League:
    def __init__(self, agent: PPOAgent):
        self.agent = agent              # the MAIN agent (never owned, never saved here)
        self.snapshots = []             # {net, rating, episode, recent}
        self.exploiters = []            # {net, episode, winrate, recent}
        self.current_rating = C.INITIAL_RATING

        self.phase = "main"             # "main" | "exploiter"
        self.exp_agent = None           # PPOAgent while an exploiter trains
        self.frozen_main = None         # frozen main actor (exploiter's opponent)
        self.main_ep_total = 0          # main-phase episodes (drives spacing)
        self.last_exploiter_ep = 0      # main_ep_total when the last phase ended
        self.phase_eps = 0              # episodes inside the current phase
        self.exp_recent = []            # exploiter scores vs frozen main (rolling)
        self.exp_gate_hit_at = None     # phase_eps when TARGET_WR was reached

    # ── opponent sampling ──────────────────────────────────────────────────────
    def sample_opponent(self):
        """Returns ("current", None) | ("snap", idx) | ("exp", idx)."""
        pool = ([("snap", j) for j in range(len(self.snapshots))]
                + [("exp", j) for j in range(len(self.exploiters))])
        r = np.random.random()
        if r < C.OPP_CURRENT_PROB or not pool:
            return "current", None
        if r < C.OPP_CURRENT_PROB + C.OPP_PFSP_PROB:
            w = np.array([self._snap_weight(self.snapshots[j]) if k == "snap"
                          else self._exp_weight(self.exploiters[j])
                          for k, j in pool], dtype=np.float64)
            i = int(np.random.choice(len(pool), p=w / w.sum()))
        else:
            i = int(np.random.randint(len(pool)))
        return pool[i]

    @staticmethod
    def _snap_weight(s):
        """PFSP weight for a past self = the main's live loss rate against it,
        floored (so solved ones never fully vanish) then raised to PFSP_POWER
        (so the few that still beat the main dominate the draw). Neutral prior
        until enough games exist, so fresh snapshots gather data first."""
        r = s["recent"]
        if len(r) >= C.PFSP_MIN_GAMES:
            return max(1.0 - sum(r) / len(r), C.PFSP_SNAP_FLOOR) ** C.PFSP_POWER
        return C.PFSP_SNAP_PRIOR ** C.PFSP_POWER

    @staticmethod
    def _exp_weight(e):
        """PFSP weight for an exploiter = live loss rate, falling back to its
        graduation winrate until enough games exist. Same floor-then-power
        shaping as _snap_weight."""
        r = e["recent"]
        if len(r) >= C.PFSP_MIN_GAMES:
            wr = 1.0 - sum(r) / len(r)
        else:
            wr = e["winrate"]
        return max(wr, C.PFSP_EXP_FLOOR) ** C.PFSP_POWER

    def opponent_net(self, kind, idx):
        if kind == "snap":
            return self.snapshots[idx]["net"]
        if kind == "exp":
            return self.exploiters[idx]["net"]
        return self.frozen_main            # "frozen" (exploiter phase)

    # ── main-phase bookkeeping ─────────────────────────────────────────────────
    def record_result(self, kind, idx, score):
        """Score is the main's result vs opponent (kind, idx): 1 win / 0.5 / 0."""
        if kind == "snap":
            item = self.snapshots[idx]
            self.current_rating, item["rating"] = elo_update(
                self.current_rating, item["rating"], score)
            item["recent"].append(score)
            del item["recent"][:-C.RECENT_CAP]
        elif kind == "exp":
            rec = self.exploiters[idx]["recent"]
            rec.append(score)
            del rec[:-C.RECENT_CAP]
        self.main_ep_total += 1

    def maybe_snapshot(self, episode_count):
        """Freeze the current main on the snapshot cadence. Returns True if the
        oldest snapshot was evicted (caller must shift in-flight snap indices)."""
        if self.main_ep_total % C.SNAPSHOT_INTERVAL != 0:
            return False
        snap_src = self.agent.actor
        self.snapshots.append({"net": self._freeze(snap_src),
                               "rating": self.current_rating,
                               "episode": episode_count, "recent": []})
        if len(self.snapshots) > C.SNAPSHOT_BUFFER:
            self.snapshots.pop(0)
            return True
        return False

    # ── pool health (exploiter trigger + logging) ──────────────────────────────
    def top_exploiter_winrate(self):
        """Highest winrate any EXPLOITER currently holds against the main (the
        trigger signal). Snapshots are deliberately excluded: a recent snapshot
        is ~the current policy and pins the max near 50% forever, which would
        make a mastery threshold unreachable. None until an exploiter has
        enough games."""
        rates = [1.0 - sum(a["recent"]) / len(a["recent"])
                 for a in self.exploiters
                 if len(a["recent"]) >= C.EXPLOITER_TRIGGER_MIN_GAMES]
        return max(rates) if rates else None

    def pool_status(self):
        """For logging: (top_wr, 'snap3'/'exp2', n_agents_beating_main)."""
        best_wr, best_id, losing = None, "-", 0
        for kind, lst in (("snap", self.snapshots), ("exp", self.exploiters)):
            for j, a in enumerate(lst):
                r = a["recent"]
                if len(r) < C.EXPLOITER_TRIGGER_MIN_GAMES:
                    continue
                wr = 1.0 - sum(r) / len(r)
                if wr > 0.5:
                    losing += 1
                if best_wr is None or wr > best_wr:
                    best_wr, best_id = wr, f"{kind}{j}"
        return best_wr, best_id, losing

    # ── exploiter phase machine ────────────────────────────────────────────────
    def should_start_exploiter(self):
        """Mastery trigger, bounded by min/max spacing: spawn once no existing
        exploiter still beats the main by EXPLOITER_TRIGGER_WR or more."""
        gap = self.main_ep_total - self.last_exploiter_ep
        if gap < C.EXPLOITER_MIN_INTERVAL:
            return False                    # let the main absorb the last one
        if gap >= C.EXPLOITER_MAX_INTERVAL:
            return True                     # diversity floor — find a new hole
        top = self.top_exploiter_winrate()
        return top is not None and top < C.EXPLOITER_TRIGGER_WR

    def start_exploiter(self):
        # frozen_main answers whole-E batches every cycle -> agent's device.
        src = self.agent.actor
        self.frozen_main = self._freeze(src, self.agent.device)
        self.exp_agent = PPOAgent(self.agent.state_dim, self.agent.num_actions,
                                  device=str(self.agent.device))
        if C.EXPLOITER_WARM_START:
            self.exp_agent.warm_start_from(self.agent)
        self.phase = "exploiter"
        self.phase_eps = 0
        self.exp_recent = []
        self.exp_gate_hit_at = None
        print(f"=== league: exploiter #{len(self.exploiters) + 1} starts "
              f"(main ep {self.main_ep_total}, "
              f"{'warm' if C.EXPLOITER_WARM_START else 'cold'} start) ===")

    def record_exp_result(self, score):
        """Score is the EXPLOITER's result vs the frozen main. Returns True if
        the phase just finished."""
        self.exp_recent.append(score)
        del self.exp_recent[:-C.EXPLOITER_WR_WINDOW]
        self.phase_eps += 1
        return self._phase_done()

    def _phase_done(self):
        """Adaptive stop: winrate gate + sharpen tail + timeout."""
        if self.phase_eps >= C.EXPLOITER_MAX_EPISODES:
            return True
        if self.phase_eps < C.EXPLOITER_MIN_EPISODES:
            return False
        if self.exp_gate_hit_at is None:
            if (len(self.exp_recent) >= C.EXPLOITER_WR_WINDOW
                    and self.exp_win_rate() >= C.EXPLOITER_TARGET_WR):
                self.exp_gate_hit_at = self.phase_eps
                print(f"=== league: exploiter hit "
                      f"{C.EXPLOITER_TARGET_WR * 100:.0f}% at {self.phase_eps} "
                      f"eps — sharpening for {C.EXPLOITER_EXTRA_EPISODES} more ===")
            return False
        return self.phase_eps >= self.exp_gate_hit_at + C.EXPLOITER_EXTRA_EPISODES

    def finish_exploiter(self, episode_count):
        """Graduate the exploiter into the pool. Returns True if the oldest
        exploiter was evicted (caller must shift in-flight exp indices)."""
        wr = self.exp_win_rate()
        exp_src = self.exp_agent.actor
        self.exploiters.append({"net": self._freeze(exp_src),
                                "episode": episode_count, "winrate": wr,
                                "recent": []})
        evicted = False
        if len(self.exploiters) > C.EXPLOITER_POOL_MAX:
            self.exploiters.pop(0)
            evicted = True
        how = "gated" if self.exp_gate_hit_at is not None else "timed out"
        print(f"=== league: exploiter #{len(self.exploiters)} done ({how} after "
              f"{self.phase_eps} eps) — {wr * 100:.1f}% vs main over its last "
              f"{len(self.exp_recent)} eps (pool {len(self.exploiters)}) ===")
        self.exp_agent = None
        self.frozen_main = None
        self.phase = "main"
        self.last_exploiter_ep = self.main_ep_total   # spacing counts from here
        return evicted

    def exploiter_entropy_coef(self):
        """EC schedule inside an exploiter phase: decay, then floor once the
        winrate gate is hit (sharpen)."""
        if self.exp_gate_hit_at is not None:
            return C.EXPLOITER_EC_END
        frac = min(1.0, self.phase_eps / C.EXPLOITER_EC_DECAY_EPISODES)
        return (C.EXPLOITER_EC_START
                + (C.EXPLOITER_EC_END - C.EXPLOITER_EC_START) * frac)

    def exp_win_rate(self):
        return (sum(self.exp_recent) / len(self.exp_recent)) if self.exp_recent else 0.0

    # ── helpers / persistence ──────────────────────────────────────────────────
    def _new_net(self):
        # Pool members are ACTORS only — plain MLPs. The prefix critic is never
        # frozen into the pool: opponents only need to act.
        from .actor import ActorNet
        return ActorNet(self.agent.state_dim, self.agent.num_actions)

    def _freeze(self, src_net, device="cpu"):
        """Default CPU: pool nets serve many SMALL per-net batches, where GPU
        dispatch overhead loses to CPU (measured ~6x worse on MPS). frozen_main
        is the exception — it sees whole-E batches, so it rides the agent's
        device."""
        net = self._new_net()
        net.load_state_dict({k: v.cpu() for k, v in src_net.state_dict().items()})
        net.to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    def serialize(self):
        """The population parts of a checkpoint (same shape as v1, so
        bonk.export_model and bonk4.play's resolve_agent work unchanged).
        An in-flight exploiter phase is NOT saved — it restarts when due."""
        return {
            "snapshots": [
                {"rating": it["rating"], "episode": it["episode"],
                 "weights": it["net"].to_records()}
                for it in self.snapshots
            ],
            "league": {
                "mainEpTotal": self.main_ep_total,
                "lastExploiterEp": self.last_exploiter_ep,
                "exploiters": [
                    {"episode": it["episode"], "winrate": it["winrate"],
                     "weights": it["net"].to_records()}
                    for it in self.exploiters
                ],
            },
        }

    def load(self, obj, episode_count):
        def frozen(records):
            # Pool nets live on CPU (see _freeze). Recurrent nets serialize as a
            # dict of named tensors, feedforward ones as a tfjs record LIST —
            # MLP.from_records cannot read the former, so dispatch on the shape
            # of the payload rather than on config (a checkpoint must load as
            # whatever it was saved as).
            if isinstance(records, dict):
                net = self._new_net()
                net.load_records(records)
            else:
                from .actor import ActorNet
            net = ActorNet.from_records(records)
            for p in net.parameters():
                p.requires_grad_(False)
            return net

        self.snapshots = [
            {"net": frozen(rec["weights"]), "rating": rec["rating"],
             "episode": rec.get("episode", 0), "recent": []}
            for rec in obj.get("snapshots", [])
        ]
        league = obj.get("league", {})
        self.exploiters = [
            {"net": frozen(rec["weights"]), "episode": rec.get("episode", 0),
             "winrate": rec.get("winrate", 0.0), "recent": []}
            for rec in league.get("exploiters", [])
        ]
        self.current_rating = obj.get("progress", {}).get(
            "currentRating", C.INITIAL_RATING)
        self.main_ep_total = league.get("mainEpTotal", episode_count)
        # Treat load as "an exploiter just ended" so min-spacing applies first.
        self.last_exploiter_ep = league.get("lastExploiterEp", self.main_ep_total)
        self.phase = "main"
        self.exp_agent = None
        self.frozen_main = None
