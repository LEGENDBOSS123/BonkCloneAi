"""Drop-in replacement for `bonk.sim.BonkSim`, backed by the real bonk engine.

`bonk2/env.py`'s LagEnv only touches a small slice of the sim:

    sim.reset(randomize_spawns)   sim.step(k0, k1)   sim.is_dead(i)
    sim.players[i].pos / .vel / .heavy_power / .is_grounded()

so presenting that surface over `bonk3.engine.BonkEngine` swaps the physics
without touching the observation, the lag model, the trainer, or STATE_DIM.
The obs stays 34-dim: `heavy_power` now comes from the engine's ability energy
(same 0..1000 range), and the masking/lastSeen features that bridge sim-to-real
are unchanged, because what `play3.mjs` can observe hasn't changed.

    from bonk3.simadapter import EngineSim
    from bonk2.env import LagEnv
    env = LagEnv(EngineSim("gang-grounds-2-0"))
"""

from __future__ import annotations

import random

# (map, spawn x, spawn y, side) -> widest survivable offset. Process-wide so the
# 300-odd envs a worker owns probe each spawn once between them.
_SAFE_GAP: dict = {}
_PROBE_OFFSETS = (8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5)

from .engine import BonkEngine
from .maps import MAP_HALF_H, MAP_HALF_W, load_map, ppm

DISC_RADIUS = 1.0          # rust step/players.rs DISC_DEFAULT_RADIUS
GROUNDED_VY_EPS = 0.35     # see EnginePlayer.is_grounded


class EnginePlayer:
    """Mirrors `bonk.sim.Player`'s read surface."""

    __slots__ = ("sim", "index", "radius")

    def __init__(self, sim: EngineSim, index: int):
        self.sim = sim
        self.index = index
        self.radius = DISC_RADIUS

    @property
    def _disc(self):
        return self.sim._state.discs[self.index]

    @property
    def pos(self):
        """Metres RELATIVE TO THE MAP CENTRE.

        The engine's own frame puts the map centre at (365/ppm, 250/ppm), but
        both the legacy sim and play3.mjs (`(px/scale - 365)/PPM`) use a
        centre-origin frame. The observation must match the deployed one, so
        the offset is removed here — otherwise every position feature is
        shifted by ~+1.2 normalized units and the policy sees pure
        out-of-distribution input.
        """
        d = self._disc
        return (d.x - self.sim.origin[0], d.y - self.sim.origin[1])

    @property
    def vel(self):
        d = self._disc
        return (d.vx, d.vy)      # a constant offset does not affect velocity

    @property
    def heavy_power(self) -> float:
        """0..1000, same scale as the old sim's heavy meter."""
        return self._disc.energy

    @property
    def heavy_active(self) -> bool:
        """Is the heavy ability actually engaged right now? `heavy_power` is
        the RESERVE, which sits at full when idle — so it cannot stand in for
        'is heavy on'."""
        return self._disc.ability_active

    def is_grounded(self) -> bool:
        """APPROXIMATION. The engine's packed readout has no grounded flag, and
        it is only consulted by the rollback view's ballistic fallback — the
        path that already runs solely inside a misprediction window, since a
        constant-key window returns the true state exactly. Near-zero vertical
        speed is a good enough stand-in there."""
        return abs(self._disc.vy) < GROUNDED_VY_EPS


class EngineSim:
    """BonkSim-compatible facade over BonkEngine."""

    def __init__(self, map_name_or_data="gang-grounds-2-0", settings=None,
                 seed: float | None = None):
        # Default to a RANDOM seed base per instance. A fixed base would give
        # every env in a process the identical seed sequence, correlating their
        # episodes (and making them outright identical under greedy actions).
        if seed is None:
            seed = float(random.randrange(1 << 30))
        self.map_data = (map_name_or_data if isinstance(map_name_or_data, dict)
                         else load_map(map_name_or_data))
        self.engine = BonkEngine(self.map_data, settings=settings, max_players=2)
        self.ppm = float(self.map_data["physics"]["pixelsPerMeter"])
        # Engine frame -> centre-origin frame (see EnginePlayer.pos). Renderers
        # working in engine coordinates must add this back.
        self.origin = (MAP_HALF_W / ppm(self.map_data),
                       MAP_HALF_H / ppm(self.map_data))
        self.players = [EnginePlayer(self, 0), EnginePlayer(self, 1)]
        self._seed = seed
        self._dead = [False, False]
        self._swap = False
        self._map_key = str(map_name_or_data)[:64]
        self.tick_count = 0
        self._state = self.engine.reset(self._teams(), seed=seed)

    def _teams(self):
        """Team 1 takes the first spawn, team 2 the second, so swapping teams
        swaps spawns. The engine ignores the seed for spawn choice, so this is
        how `randomize_spawns` is expressed."""
        return [(0, 2), (1, 1)] if self._swap else [(0, 1), (1, 2)]

    # ── BonkSim surface ────────────────────────────────────────────────────
    def reset(self, randomize_spawns: bool = True, spawn_frac: float = 1.0,
              close_m: float = 5.0, jitter_m: float = 0.0, positions=None):
        """spawn_frac interpolates the SECOND disc between a close start and
        its natural spawn: 0.0 = `close_m` metres from disc 0, 1.0 = untouched
        (the map's own spawn). Anything >= 1 costs nothing — the fast path is
        the plain engine reset with no state round-trip.

        Disc 0 is never moved, so one player always stands on real spawn
        ground; disc 1 is placed at the SAME HEIGHT, offset horizontally toward
        where it would naturally have been. Same height is what keeps both on
        one platform instead of dropping disc 1 into a pit.
        """
        if randomize_spawns:
            self._swap = random.random() < 0.5
        self._seed += 1.0
        self._dead = [False, False]
        self.tick_count = 0
        self._state = self.engine.reset(self._teams(), seed=self._seed)
        if positions is not None:
            self._place_discs(positions)
        elif spawn_frac < 1.0:
            self._apply_spawn_curriculum(spawn_frac, close_m, jitter_m)

    def _place_discs(self, positions):
        """Put both discs at explicit ((x0,y0),(x1,y1)) ground points at rest.
        Positions come from a pre-probed survivable pool (see ppo6.gen_spawns),
        so neither disc starts in a death trap."""
        st = self.engine.state_json()
        d = st.get("discs") or []
        for i, (x, y) in enumerate(positions):
            if i >= len(d):
                break
            d[i]["x"], d[i]["y"] = float(x), float(y)
            for k in ("velocityX", "velocityY"):
                if k in d[i]:
                    d[i][k] = 0.0
            if "spawnX" in d[i]:
                d[i]["spawnX"], d[i]["spawnY"] = float(x), float(y)
        self.engine.set_state_json(st)
        self._state = self.engine.snapshot()

    def _apply_spawn_curriculum(self, frac, close_m, jitter_m):
        """Mixture, not interpolation: with probability `frac` leave the map's
        own spawns alone, otherwise place disc 1 close to disc 0.

        Interpolating the position was the obvious approach and it is wrong —
        a point part-way between two spawns is usually mid-air over a pit, and
        measured 0% survival at frac 0.25-0.75 on a hazard map. Both ENDPOINTS
        are known-good ground, so sampling between the two task distributions
        keeps every start safe while shifting the mix exactly the same way.
        """
        if random.random() < frac:
            return                                # natural spawn this episode
        st = self.engine.state_json()
        d = st.get("discs") or []
        if len(d) < 2:
            return
        x0, y0 = d[0]["x"], d[0]["y"]
        dx = d[1]["x"] - x0
        if abs(dx) < 1e-6:
            return
        side = 1.0 if dx > 0 else -1.0
        want = close_m + (random.uniform(-jitter_m, jitter_m) if jitter_m else 0.0)
        gap = self._safe_gap(x0, y0, side, max(1.2, want))
        if gap <= 0.0:
            return                                # nowhere safe: leave it
        d[1]["x"] = x0 + side * gap
        d[1]["y"] = y0                            # same height => same platform
        for k in ("velocityX", "velocityY"):
            if k in d[1]:
                d[1][k] = 0.0
        # spawnX/spawnY drive respawns; keep them consistent so a mid-round
        # respawn cannot silently undo the curriculum.
        if "spawnX" in d[1]:
            d[1]["spawnX"], d[1]["spawnY"] = d[1]["x"], d[1]["y"]
        self.engine.set_state_json(st)
        self._state = self.engine.snapshot()

    def _safe_gap(self, x0, y0, side, want):
        """Largest offset <= `want` at which disc 1 does not immediately die.

        Platform widths differ per spawn — measured 100% survival at 2 m but
        50% at 3 m on `death`, because one spawn sits on a narrow slab. So probe
        it rather than guessing, once per (spawn, side), cached process-wide and
        shared by every env in this worker.
        """
        key = (self._map_key, round(x0, 1), round(y0, 1), side)
        hit = _SAFE_GAP.get(key)
        if hit is None:
            hit = self._probe_safe(x0, y0, side)
            _SAFE_GAP[key] = hit
        return min(want, hit)

    def _probe_safe(self, x0, y0, side, frames=60):
        """Descending scan for the widest survivable offset. Destroys the live
        state, so the caller's reset is replayed afterwards."""
        best = 0.0
        for cand in _PROBE_OFFSETS:
            # RESET first. Reading state_json() straight into the next
            # candidate inherits the previous one's post-simulation state —
            # including a disc that just died — so every candidate after the
            # first was being judged from a corpse.
            self.engine.reset(self._teams(), seed=self._seed)
            st = self.engine.state_json()
            d = st.get("discs") or []
            if len(d) < 2:
                break
            d[1]["x"], d[1]["y"] = d[0]["x"] + side * cand, d[0]["y"]
            for k in ("velocityX", "velocityY"):
                d[1][k] = 0.0
            self.engine.set_state_json(st)
            res = None
            for _ in range(frames):
                res = self.engine.step([None, None])
            if res and res.discs[0].present and res.discs[1].present:
                best = cand
                break
        # the probe trashed the round; put the caller's reset back
        self._state = self.engine.reset(self._teams(), seed=self._seed)
        return best

    def step(self, keys0: dict, keys1: dict):
        self._state = self.engine.step([_to_input(keys0), _to_input(keys1)])
        for idx, _kind in self._state.deaths:
            if 0 <= idx < 2:
                self._dead[idx] = True
        # A disc that vanishes from the readout is also dead.
        for i in range(2):
            if not self._state.discs[i].present:
                self._dead[i] = True
        self.tick_count += 1

    def is_dead(self, i: int) -> bool:
        return self._dead[i]

    # ── fork / rewind (lookahead search) ────────────────────────────────────
    # `engine.state_json()`/`set_state_json()` round-trip the PHYSICS state
    # only. `EngineSim` layers its OWN cache on top -- `_state` (the fast-path
    # decoded readout `EnginePlayer.pos`/`.vel`/etc actually read) and `_dead`
    # (derived from a STEP TRANSITION, not the state itself, so it does not
    # follow from the physics alone). A caller that restores physics via
    # `set_state_json` directly and skips these leaves both stale: positions
    # silently keep reading whatever the last real step() produced, and a
    # death flag set by a discarded speculative rollout keeps reporting dead
    # after the "real" state is restored alive. `get_full_state`/
    # `set_full_state` are the pair that actually round-trips everything this
    # wrapper is responsible for.
    def get_full_state(self):
        return {"phys": self.engine.state_json(), "dead": list(self._dead),
               "tick_count": self.tick_count}

    def set_full_state(self, saved) -> None:
        self.engine.set_state_json(saved["phys"])
        self._state = self.engine.snapshot()   # refresh the cached fast-path readout
        self._dead = list(saved["dead"])
        self.tick_count = saved["tick_count"]

    # ── extras (rendering / debugging) ─────────────────────────────────────
    @property
    def frozen(self) -> bool:
        return self._state.frozen

    def close(self):
        self.engine.close()


def _to_input(keys: dict) -> dict:
    """bonk.sim key dict -> engine input dict ('heavy' is the engine's 'action')."""
    return {
        "up": keys.get("up", False),
        "down": keys.get("down", False),
        "left": keys.get("left", False),
        "right": keys.get("right", False),
        "action": keys.get("heavy", False),
    }


def make_sim(map_json_or_name=None, engine: str = "real", settings=None):
    """Factory shared by the collector, eval and play so they cannot disagree.

    engine="real"   -> EngineSim (bit-identical bonk.io physics)
    engine="legacy" -> bonk.sim.BonkSim (the old Box2D port of our JS clone)
    """
    if engine == "legacy":
        import json

        from bonk.sim import BonkSim
        if isinstance(map_json_or_name, dict):
            return BonkSim(map_json_or_name)
        with open(map_json_or_name) as f:
            return BonkSim(json.load(f))
    return EngineSim(map_json_or_name or "gang-grounds-2-0", settings=settings)
