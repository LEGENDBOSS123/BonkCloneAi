"""AlphaStar-lite population: snapshots, PFSP sampling, and exploiter phases.

One module owns everything population-related so the trainer stays a pure
PPO/collection loop.

* **snapshots** — frozen past selves, taken on a step cadence.
* **exploiters** — graduated best-responses to a FROZEN main. A phase starts
  once the main has MASTERED its pool (no existing exploiter still beats it by
  `trigger_wr`), subject to min spacing. During a phase, main training pauses
  and the exploiter trains against the frozen main; the phase ends on a winrate
  gate plus a sharpen tail, or on timeout.
* **opponent sampling** (main phase) — a mirror match against the live model, a
  PFSP draw over the pool weighted by the main's live LOSS rate (whoever beats
  the main most gets the most games), and a uniform slice that keeps loss rates
  fresh against "solved" opponents.

All spacing is driven by STEPS, never episodes: episode length varies by more
than 10x over training, so an episode-based interval fires at a rate set by how
the agent happens to be playing rather than by experience collected.

Unlike `ppo7/league.py` this module does not import the agent class. It takes
factories instead, which breaks the `league -> ppo -> league` import cycle and
lets the phase machine be tested without torch.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

from .config import LeagueConfig
from .mingru import MinGRUNet, Records
from .nets import freeze, net_from_records


class AgentLike(Protocol):
    """The slice of `PPOAgent` the league actually touches."""

    actor: MinGRUNet
    critic: MinGRUNet
    state_dim: int
    num_actions: int
    device: Any


AgentFactory = Callable[..., AgentLike]

Kind = str          # "current" | "snap" | "exp" | "frozen" | "idle"


def elo_update(ra: float, rb: float, score_a: float,
               k: float) -> tuple[float, float]:
    """Standard Elo, returning both updated ratings."""
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


class League:
    """The opponent pool and the exploiter phase machine."""

    def __init__(self, agent: AgentLike, cfg: LeagueConfig,
                 agent_factory: AgentFactory,
                 rng: np.random.Generator | None = None) -> None:
        """
        Args:
            agent:         the MAIN agent — never owned here, never saved here.
            cfg:           league settings, including the exploiter sub-config.
            agent_factory: builds an exploiter. Called as
                           ``agent_factory(atoms=..., is_exploiter=True,
                           is_staller=...)``.
            rng:           sampling stream; a fresh default when omitted.
        """
        self.agent = agent
        self.cfg = cfg
        self.exp_cfg = cfg.exploiter
        self._make_agent = agent_factory
        self.rng = rng if rng is not None else np.random.default_rng()

        self.snapshots: list[dict[str, Any]] = []   # {net, rating, episode, recent}
        self.exploiters: list[dict[str, Any]] = []  # {net, episode, winrate, staller, recent}
        self.current_rating = cfg.initial_rating

        self.phase = "main"                         # "main" | "exploiter"
        self.exp_agent: AgentLike | None = None
        self.frozen_main: MinGRUNet | None = None
        self.exp_is_staller = False
        self.exp_ec_start = self.exp_cfg.ec_start

        self.main_ep_total = 0                      # reporting only
        self.main_steps = 0
        self.last_snapshot_steps = 0
        self.last_exploiter_ep = 0                  # reporting only
        self.last_exploiter_steps = 0
        self.phase_eps = 0
        self.phase_steps = 0
        self.exp_recent: list[float] = []
        self.exp_gate_hit_at: int | None = None

        # The main's results against the two FIXED categories. `record_result`
        # already saw these and threw them away; scope="all" needs them to know
        # what the idle and self-play shares contribute to the aggregate.
        self.idle_recent: list[float] = []
        self.cur_recent: list[float] = []
        # Cached calibration (see `_calibration`); pool winrates move far more
        # slowly than the per-episode rate this would otherwise be solved at.
        self._calib: dict[str, Any] | None = None
        self._calib_calls = 0
        # Bumped on every pool MUTATION. Length alone is not enough to detect
        # one: at `snapshot_buffer` members `maybe_snapshot` pops the oldest and
        # appends, so in steady state the length never changes while every
        # index shifts down by one — a cached probability vector would then be
        # silently applied to a different pool.
        self._pool_version = 0

    # ── opponent sampling ──────────────────────────────────────────────────
    def sample_opponent(self) -> tuple[Kind, int | None]:
        """Draw this env's next opponent.

        Returns one of ``("idle", None)``, ``("current", None)``,
        ``("snap", i)``, ``("exp", i)``.
        """
        c = self.cfg
        if self.rng.random() < c.opp_idle_prob:
            return "idle", None                     # scripted do-nothing dummy
        pool = ([("snap", j) for j in range(len(self.snapshots))]
                + [("exp", j) for j in range(len(self.exploiters))])
        r = self.rng.random()
        if r < c.opp_current_prob or not pool:
            return "current", None
        if r < c.opp_current_prob + c.opp_pfsp_prob:
            i = int(self.rng.choice(len(pool), p=self._pfsp_probs(pool)))
        else:
            i = int(self.rng.integers(len(pool)))   # keeps stale loss rates fresh
        return pool[i]

    # ── PFSP weights -> calibrated sampling probabilities ──────────────────
    def _base_weights(self, pool: list[tuple[str, int]]) -> np.ndarray:
        """The EXISTING per-opponent PFSP weights, unchanged."""
        return np.array([self._snap_weight(self.snapshots[j]) if k == "snap"
                         else self._exp_weight(self.exploiters[j])
                         for k, j in pool], dtype=np.float64)

    def _est_wr(self, kind: str, item: dict[str, Any]) -> float:
        """The MAIN's estimated winrate against one pool member.

        This is EXACTLY the number `_snap_weight`/`_exp_weight` feed into
        `_pfsp` whenever they have enough games to use one. Below
        `pfsp_min_games` those use a flat prior WEIGHT rather than a winrate,
        so there is no value to reuse; the few observed games are shrunk toward
        the same prior they assume, with `pfsp_min_games` as the prior
        strength. A 2-game 100% therefore reads ~0.52, not 1.0 — a low-sample
        opponent can never yank the calibration around.
        """
        r = item["recent"]
        n = len(r)
        if n >= self.cfg.pfsp_min_games:
            return float(sum(r)) / n
        k = float(self.cfg.pfsp_min_games)
        prior = 0.5 if kind == "snap" else 1.0 - float(item["winrate"])
        return (float(sum(r)) + prior * k) / (n + k)

    def _fixed_wr(self, recent: list[float]) -> float:
        """Prior-shrunk winrate for a FIXED category (idle / current)."""
        k = float(self.cfg.pfsp_min_games)
        return (float(sum(recent)) + 0.5 * k) / (len(recent) + k)

    @staticmethod
    def _probs_from_logits(z: np.ndarray, floor_mix: float) -> np.ndarray:
        """softmax(z), then mixed with a uniform floor.

        The max-subtraction is what keeps exp() in range at any tilt, so no
        `lam` can produce an inf or a NaN here. The uniform mixture afterwards
        guarantees every member keeps `p >= floor_mix/n` however hard the tilt
        pushes — the diversity guarantee, and a real probability floor rather
        than the unnormalised weight floors `_pfsp` applies.
        """
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        if floor_mix > 0.0:
            p = (1.0 - floor_mix) * p + floor_mix / len(p)
        return p

    def _tilt_logits(self, w: np.ndarray, q: np.ndarray, lam: float) -> np.ndarray:
        """`log(base_weight) - lam * wr`.

        This is `w_i * exp(lam * (target - q_i))` with the `exp(lam*target)`
        factor dropped: it is constant across i and cancels in the
        normalisation, so carrying it would add nothing but a way to overflow.
        """
        return np.log(np.maximum(w, 1e-300)) - lam * q

    def _solve_tilt(self, w: np.ndarray, q: np.ndarray, target: float
                    ) -> tuple[float, float]:
        """Find `lam` with E_p[q] == target, by bisection. Returns (lam, achieved).

        E_p[q] is strictly DECREASING in `lam` (its derivative is -Var_p(q)),
        so bisection on [-max_tilt, +max_tilt] converges and cannot cycle. If
        the target lies outside what that bracket reaches — no distribution
        over these opponents can average it, or the uniform floor holds mass on
        the wrong side — the nearest reachable endpoint is returned rather than
        pushing the tilt toward infinity.
        """
        fm = self.cfg.sampled_wr_floor_mix

        def f(lam: float) -> float:
            return float(self._probs_from_logits(
                self._tilt_logits(w, q, lam), fm) @ q)

        L = float(self.cfg.sampled_wr_max_tilt)
        lo, hi = -L, L                          # f(lo) is the MAX, f(hi) the MIN
        f_lo, f_hi = f(lo), f(hi)
        if target >= f_lo:
            return lo, f_lo
        if target <= f_hi:
            return hi, f_hi
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        return lam, f(lam)

    def _calibration(self, pool: list[tuple[str, int]]) -> dict[str, Any]:
        """Solve (or reuse) the tilt for the current pool. See `ArchiveConfig`-
        style notes on `LeagueConfig.calibrate_sampled_wr` for the why."""
        c = self.cfg
        w = self._base_weights(pool)
        q = np.array([self._est_wr(k, self.snapshots[j] if k == "snap"
                                   else self.exploiters[j])
                      for k, j in pool], dtype=np.float64)
        base_p = w / w.sum()
        out: dict[str, Any] = {
            "n": len(pool), "version": self._pool_version, "q": q, "base_p": base_p,
            "min_wr": float(q.min()), "max_wr": float(q.max()),
            "base_sampled_wr": float(base_p @ q),
        }
        if not c.calibrate_sampled_wr or len(pool) < 2:
            out.update(p=base_p, lam=0.0, target=float("nan"),
                       eff_target=float("nan"), sampled_wr=float(base_p @ q))
            return out

        # Fixed categories keep their configured probabilities; scope="all"
        # only changes what the PFSP slice must supply so the TOTAL lands on
        # target. a_* are the true unconditional category probabilities: the
        # idle draw is an independent Bernoulli BEFORE the r-draw, so the other
        # three are all scaled by (1 - opp_idle_prob).
        tgt = float(c.target_sampled_wr)
        rest = 1.0 - c.opp_idle_prob
        a_idle = c.opp_idle_prob
        a_cur = rest * c.opp_current_prob
        a_pfsp = rest * c.opp_pfsp_prob
        a_uni = rest * max(0.0, 1.0 - c.opp_current_prob - c.opp_pfsp_prob)
        if c.sampled_wr_scope == "all" and a_pfsp > 1e-9:
            q_idle, q_cur = self._fixed_wr(self.idle_recent), self._fixed_wr(self.cur_recent)
            fixed = a_idle * q_idle + a_cur * q_cur + a_uni * float(q.mean())
            pool_target = (tgt - fixed) / a_pfsp
            out.update(q_idle=q_idle, q_cur=q_cur, a_idle=a_idle, a_cur=a_cur,
                       a_pfsp=a_pfsp, a_uni=a_uni)
        else:
            pool_target = tgt

        # Clamp to what ANY distribution over these opponents could average,
        # before the solver ever sees it, so an impossible ask is reported as a
        # clamp rather than as a silent max-tilt.
        eff = min(max(pool_target, float(q.min())), float(q.max()))
        lam, achieved = self._solve_tilt(w, q, eff)
        p = self._probs_from_logits(self._tilt_logits(w, q, lam),
                                    c.sampled_wr_floor_mix)
        out.update(p=p, lam=float(lam), target=tgt, pool_target=float(pool_target),
                   eff_target=float(eff), sampled_wr=float(p @ q),
                   achieved=float(achieved))
        return out

    def _pfsp_probs(self, pool: list[tuple[str, int]]) -> np.ndarray:
        """Final normalised sampling probabilities over the PFSP pool."""
        self._calib_calls += 1
        stale = (self._calib is None
                 or self._calib["n"] != len(pool)
                 or self._calib["version"] != self._pool_version
                 or self._calib_calls % max(1, self.cfg.sampled_wr_recalc_every) == 0)
        if stale:
            self._calib = self._calibration(pool)
        return self._calib["p"]

    def sampling_diagnostics(self) -> dict[str, Any] | None:
        """Everything needed to debug the sampler, or None outside a pool."""
        pool = ([("snap", j) for j in range(len(self.snapshots))]
                + [("exp", j) for j in range(len(self.exploiters))])
        if not pool:
            return None
        d = dict(self._calibration(pool))
        p, q = d["p"], d["q"]
        c = self.cfg
        rest = 1.0 - c.opp_idle_prob
        a_pfsp = rest * c.opp_pfsp_prob
        a_uni = rest * max(0.0, 1.0 - c.opp_current_prob - c.opp_pfsp_prob)
        # The number the user actually cares about: expected WR of the ACTUAL
        # training matchup distribution, fixed categories included.
        d["overall_wr"] = float(
            c.opp_idle_prob * self._fixed_wr(self.idle_recent)
            + rest * c.opp_current_prob * self._fixed_wr(self.cur_recent)
            + a_uni * float(q.mean())
            + a_pfsp * float(p @ q))
        nz = p[p > 0]
        d["entropy"] = float(-(nz * np.log(nz)).sum())
        d["max_entropy"] = float(np.log(len(p)))
        top = np.argsort(-p)[:5]
        d["top"] = [(pool[i][0] + str(pool[i][1]), float(p[i]), float(q[i])) for i in top]
        d["min_p"] = float(p.min())
        return d

    def _pfsp(self, main_wr: float) -> float:
        """PFSP curve over the MAIN's winrate against an opponent.

        ``hard`` concentrates play on the opponents the main LOSES to — the
        main-killers that keep it engaged — raised to `pfsp_power` so the
        weighting is genuinely peaked rather than linear. ``even`` peaks at
        `pfsp_target_wr`, so the pool is dominated by matchups that are still
        winnable rather than by exploiters that have already solved the main.
        ``var`` is AlphaStar's ``x(1-x)``.
        """
        c = self.cfg
        if c.pfsp_mode == "var":
            # return float(main_wr * (1.0 - main_wr))
            return float((main_wr ** 1.5) * (1.0 - main_wr)) # maxes out at 0.6 instead of 0.5, so the main gets more games against its killers
        if c.pfsp_mode == "hard":
            return float((1.0 - main_wr) ** c.pfsp_power)
        d = main_wr - c.pfsp_target_wr
        return float(np.exp(-(d * d) / (2.0 * c.pfsp_sigma ** 2)))

    def _snap_weight(self, s: dict[str, Any]) -> float:
        """Weight for a past self: the main's live loss rate against it.

        Floored so a solved snapshot never fully vanishes, with a neutral prior
        until enough games exist so fresh snapshots gather data first.
        """
        r = s["recent"]
        if len(r) >= self.cfg.pfsp_min_games:
            return max(self._pfsp(sum(r) / len(r)), self.cfg.pfsp_snap_floor)
        return self.cfg.pfsp_snap_prior

    def _exp_weight(self, e: dict[str, Any]) -> float:
        """Weight for an exploiter: live loss rate, falling back to its
        graduation winrate until enough games exist."""
        r = e["recent"]
        if len(r) >= self.cfg.pfsp_min_games:
            main_wr = sum(r) / len(r)
        else:
            main_wr = 1.0 - e["winrate"]        # stored value is the EXPLOITER's
        return max(self._pfsp(main_wr), self.cfg.pfsp_exp_floor)

    def opponent_net(self, kind: Kind, idx: int | None) -> MinGRUNet:
        """The frozen net for a sampled opponent."""
        if kind == "snap":
            return self.snapshots[idx]["net"]
        if kind == "exp":
            return self.exploiters[idx]["net"]
        assert self.frozen_main is not None, "no frozen main outside an exploiter phase"
        return self.frozen_main

    def is_staller(self, kind: Kind, idx: int | None) -> bool:
        """True if this opponent is a graduated STALLER exploiter."""
        return (kind == "exp" and idx is not None
                and 0 <= idx < len(self.exploiters)
                and bool(self.exploiters[idx].get("staller", False)))

    # ── main-phase bookkeeping ─────────────────────────────────────────────
    def add_steps(self, n: int) -> None:
        """Advance the clocks. `main_steps` paces the league; `phase_steps`
        gates the current exploiter."""
        if self.phase == "exploiter":
            self.phase_steps += n
        else:
            self.main_steps += n

    def record_result(self, kind: Kind, idx: int | None, score: float,
                      elo_score: float | None = None) -> None:
        """Record the MAIN's result against ``(kind, idx)``.

        `score` is the atom-scored result (`Trainer._wr_score`) and drives the
        winrate stats. `elo_score` is the chess 1/0.5/0 convention Elo is
        defined for — a rating system fed "a draw is a loss" would drift for a
        reason that has nothing to do with strength. Defaults to `score` for
        callers that have only one.
        """
        elo = score if elo_score is None else elo_score
        if kind == "snap":
            item = self.snapshots[idx]
            self.current_rating, item["rating"] = elo_update(
                self.current_rating, item["rating"], elo, self.cfg.elo_k)
            item["recent"].append(score)
            del item["recent"][:-self.cfg.recent_cap]
        elif kind == "exp":
            rec = self.exploiters[idx]["recent"]
            rec.append(score)
            del rec[:-self.cfg.recent_cap]
        elif kind == "idle":
            # Previously dropped on the floor. `sampled_wr_scope="all"` needs
            # to know what the idle share contributes to the aggregate rather
            # than assuming it.
            self.idle_recent.append(score)
            del self.idle_recent[:-self.cfg.recent_cap]
        elif kind == "current":
            self.cur_recent.append(score)
            del self.cur_recent[:-self.cfg.recent_cap]
        self.main_ep_total += 1

    def maybe_snapshot(self, episode_count: int) -> bool:
        """Freeze the current main on the step cadence.

        Returns True if the oldest snapshot was EVICTED, so the caller can
        shift any in-flight opponent indices down by one.
        """
        if not self.cfg.enabled:
            return False
        if self.main_steps - self.last_snapshot_steps < self.cfg.snapshot_interval_steps:
            return False
        self.last_snapshot_steps = self.main_steps
        self._pool_version += 1
        self.snapshots.append({"net": freeze(self.agent.actor),
                               "rating": self.current_rating,
                               "episode": episode_count, "recent": []})
        if len(self.snapshots) > self.cfg.snapshot_buffer:
            self.snapshots.pop(0)
            return True
        return False

    # ── pool health ────────────────────────────────────────────────────────
    def top_exploiter_winrate(self) -> float | None:
        """Highest winrate any EXPLOITER currently holds against the main.

        Snapshots are deliberately excluded: a recent snapshot is ~the current
        policy, which pins the maximum near 50% forever and would make the
        mastery threshold unreachable. None until an exploiter has enough games.
        """
        rates = [1.0 - sum(a["recent"]) / len(a["recent"])
                 for a in self.exploiters
                 if len(a["recent"]) >= self.exp_cfg.trigger_min_games
                 and not self._excluded_from_top(a)]
        return max(rates) if rates else None

    def _excluded_from_top(self, entry: dict[str, Any]) -> bool:
        """Whether a pool entry is kept out of the MAXIMUM (not out of
        sampling, training, or the losing tally). See
        `LeagueConfig.topwr_exclude_stallers`."""
        return (self.cfg.topwr_exclude_stallers
                and bool(entry.get("staller", False)))

    def pool_status(self) -> tuple[float | None, str, int]:
        """For the log line: ``(top_winrate_vs_main, "snap3", n_beating_main)``.

        `losing` deliberately still counts stallers even when the max excludes
        them: "how many opponents beat me" is real information, and a run whose
        `topWR` reads `--` while `lose` is nonzero is telling you the only
        things beating the main right now are turtles.
        """
        best_wr, best_id, losing = None, "-", 0
        for kind, lst in (("snap", self.snapshots), ("exp", self.exploiters)):
            for j, a in enumerate(lst):
                r = a["recent"]
                if len(r) < self.exp_cfg.trigger_min_games:
                    continue
                wr = 1.0 - sum(r) / len(r)
                if wr > 0.5:
                    losing += 1
                if self._excluded_from_top(a):
                    continue
                if best_wr is None or wr > best_wr:
                    best_wr, best_id = wr, f"{kind}{j}"
        return best_wr, best_id, losing

    # ── exploiter phase machine ────────────────────────────────────────────
    def should_start_exploiter(self) -> bool:
        """Mastery trigger, bounded by min spacing."""
        if not self.cfg.enabled:
            return False
        gap = self.main_steps - self.last_exploiter_steps
        if gap < self.exp_cfg.min_interval_steps:
            return False                    # let the main absorb the last one
        top = self.top_exploiter_winrate()
        if top is None:
            # Pool empty, or nothing has enough recent games to judge. Bootstrap
            # the first probe ONLY when the pool is truly empty, so a
            # transiently-unmeasured pool cannot bypass the mastery gate.
            return not self.exploiters
        mastered = top < self.exp_cfg.trigger_wr
        if self.exp_cfg.require_mastery:
            # Open a new hole only once the main has mastered the pool. If it
            # cannot, piling on more exploiters just walls it off further — a
            # measured death spiral — so keep training against the pool instead.
            return mastered
        return mastered or gap >= self.exp_cfg.max_interval_steps

    def start_exploiter(self, force_staller: bool | None = None,
                        quiet: bool = False) -> None:
        """Begin an exploiter phase against a freshly frozen main.

        Args:
            force_staller: None takes the normal random draw; True/False pins
                the type, for deliberately testing one (``--exploiter-type``).
            quiet: suppress the banner (tests).
        """
        exp = self.exp_cfg
        # frozen_main answers whole-E batches every cycle -> keep it on the
        # agent's own device, unlike the CPU-resident pool.
        self.frozen_main = freeze(self.agent.actor, self.agent.device)

        # Decide the TYPE first: a staller is rewarded for drawing the frozen
        # main, so its critic needs the staller atom set.
        self.exp_is_staller = (bool(force_staller) if force_staller is not None
                               else bool(self.rng.random() < exp.staller_prob))
        atoms = exp.staller_atoms if self.exp_is_staller else exp.atoms
        self.exp_agent = self._make_agent(atoms=atoms, is_exploiter=True,
                                          is_staller=self.exp_is_staller)

        # Per-exploiter starting entropy: a log-uniform draw, i.e. a population
        # search over explore/exploit rather than one bet.
        if exp.ec_random:
            lo, hi = exp.ec_start_min, exp.ec_start_max
            self.exp_ec_start = float(np.exp(
                self.rng.uniform(np.log(lo), np.log(hi))))
        else:
            self.exp_ec_start = exp.ec_start

        seed_from = "current main"
        if exp.warm_start:
            src = self.agent.actor
            if not exp.seed_from_current and exp.seed_from_history and self.snapshots:
                n = len(self.snapshots)     # index -1 is newest
                w = np.array([exp.seed_decay ** (n - 1 - i) for i in range(n)],
                             dtype=np.float64)
                j = int(self.rng.choice(n, p=w / w.sum()))
                src = self.snapshots[j]["net"]
                seed_from = f"snapshot {j + 1}/{n}"
            self.exp_agent.actor.load_state_dict(
                {k: v.to(self.exp_agent.device) for k, v in src.state_dict().items()})
            # The critic always starts from the LIVE one: a snapshot's critic is
            # calibrated to a different opponent distribution, and the exploiter
            # needs a value function for the CURRENT frozen main.
            self.exp_agent.critic.load_state_dict(self.agent.critic.state_dict())

        self.phase = "exploiter"
        self.phase_eps = 0
        self.phase_steps = 0
        self.exp_recent = []
        self.exp_gate_hit_at = None
        if not quiet:
            print(f"=== league: exploiter #{len(self.exploiters) + 1} starts "
                  f"(main ep {self.main_ep_total}, "
                  f"{'warm from ' + seed_from if exp.warm_start else 'cold'}"
                  f"{', STALLER' if self.exp_is_staller else ''}"
                  f", ec0={self.exp_ec_start:.3f}) ===")

    def record_exp_result(self, score: float) -> bool:
        """Record the EXPLOITER's result vs the frozen main.

        Returns True if the phase just finished.
        """
        self.exp_recent.append(score)
        del self.exp_recent[:-self.exp_cfg.wr_window]
        self.phase_eps += 1
        return self._phase_done()

    def _phase_done(self) -> bool:
        """Adaptive stop: winrate gate, then a sharpen tail; timeout otherwise."""
        exp = self.exp_cfg
        if self.phase_steps >= exp.max_steps:
            return True
        if self.phase_steps < exp.min_steps:
            return False
        if self.exp_gate_hit_at is None:
            if (len(self.exp_recent) >= exp.wr_window
                    and self.exp_win_rate() >= exp.target_wr):
                self.exp_gate_hit_at = self.phase_steps
                print(f"=== league: exploiter hit {exp.target_wr * 100:.0f}% at "
                      f"{self.phase_eps} eps — sharpening for "
                      f"{exp.extra_steps} more steps ===")
            return False
        return self.phase_steps >= self.exp_gate_hit_at + exp.extra_steps

    def finish_exploiter(self, episode_count: int, quiet: bool = False) -> bool:
        """Graduate the exploiter into the pool.

        Returns True if the oldest exploiter was EVICTED.
        """
        wr = self.exp_win_rate()
        assert self.exp_agent is not None
        self._pool_version += 1
        self.exploiters.append({"net": freeze(self.exp_agent.actor),
                                "episode": episode_count, "winrate": wr,
                                "staller": bool(self.exp_is_staller),
                                "recent": []})
        evicted = False
        if len(self.exploiters) > self.exp_cfg.pool_max:
            self.exploiters.pop(0)
            evicted = True
        if not quiet:
            how = "gated" if self.exp_gate_hit_at is not None else "timed out"
            print(f"=== league: exploiter #{len(self.exploiters)} done ({how} "
                  f"after {self.phase_eps} eps) — {wr * 100:.1f}% vs main over "
                  f"its last {len(self.exp_recent)} eps "
                  f"(pool {len(self.exploiters)}) ===")
        self.exp_agent = None
        self.frozen_main = None
        self.phase = "main"
        self.last_exploiter_ep = self.main_ep_total
        self.last_exploiter_steps = self.main_steps     # spacing counts from here
        return evicted

    def exp_win_rate(self) -> float:
        """The exploiter's rolling winrate against the frozen main."""
        return (sum(self.exp_recent) / len(self.exp_recent)) if self.exp_recent else 0.0

    # ── persistence ────────────────────────────────────────────────────────
    def serialize(self) -> dict[str, Any]:
        """The population half of a checkpoint.

        An in-flight exploiter phase is deliberately NOT saved: it restarts
        when due, rather than resuming a half-trained best-response against a
        main that has since moved.
        """
        return {
            "snapshots": [{"rating": it["rating"], "episode": it["episode"],
                           "weights": it["net"].to_records()}
                          for it in self.snapshots],
            "league": {
                "mainEpTotal": self.main_ep_total,
                "mainSteps": self.main_steps,
                "lastSnapshotSteps": self.last_snapshot_steps,
                "lastExploiterEp": self.last_exploiter_ep,
                "lastExploiterSteps": self.last_exploiter_steps,
                "exploiters": [{"episode": it["episode"], "winrate": it["winrate"],
                                "staller": it.get("staller", False),
                                "weights": it["net"].to_records()}
                               for it in self.exploiters],
            },
        }

    def load(self, obj: dict[str, Any], episode_count: int) -> None:
        """Restore the pool. Always resumes in the MAIN phase."""
        def frozen(records: Records) -> MinGRUNet:
            net = net_from_records(records)     # pool nets live on CPU
            for p in net.parameters():
                p.requires_grad_(False)
            net.eval()
            return net

        self.snapshots = [{"net": frozen(rec["weights"]), "rating": rec["rating"],
                           "episode": rec.get("episode", 0), "recent": []}
                          for rec in obj.get("snapshots", [])]
        league = obj.get("league", {})
        self.exploiters = [{"net": frozen(rec["weights"]),
                            "episode": rec.get("episode", 0),
                            "winrate": rec.get("winrate", 0.0),
                            "staller": rec.get("staller", False), "recent": []}
                           for rec in league.get("exploiters", [])]
        self.current_rating = obj.get("progress", {}).get(
            "currentRating", self.cfg.initial_rating)
        self.main_ep_total = league.get("mainEpTotal", episode_count)
        self.main_steps = league.get("mainSteps", 0)
        self.last_snapshot_steps = league.get("lastSnapshotSteps", self.main_steps)
        # Treat a load as "an exploiter just ended", so min spacing applies first.
        self.last_exploiter_ep = league.get("lastExploiterEp", self.main_ep_total)
        self.last_exploiter_steps = league.get("lastExploiterSteps", self.main_steps)
        self.phase = "main"
        self.exp_agent = None
        self.frozen_main = None
        # The pool changed wholesale; any cached tilt describes the old one.
        self._calib = None
        self._calib_calls = 0
