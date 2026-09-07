"""ctypes binding to bonk-enviroment's Rust engine — the REAL bonk.io physics.

Why this exists: `bonk/sim.py` is a Box2D port of our own JS clone, verified to
~1e-5 against that clone. `bonk-recreation/bonk-enviroment` is a bit-identical
reimplementation of the actual game, and it ships a Rust core with a plain C ABI
(`benv_*`), so Python can drive it directly with no JS runtime and no per-frame
serialization — per-frame traffic is flat f64 buffers.

Measured on the map we train on (Gang Grounds 2.0, 22 shapes): ~198k
env-steps/s at 320 envs, i.e. slightly FASTER than the current Python+Box2D sim
(~184k) while being bit-exact with the real game. Heavier maps cost more
(food fight, 83 shapes + joints: ~12k/s) — that is map complexity, not engine
overhead.

Build the library first:  python/bonk3/build.sh

    from bonk3.engine import BonkEngine
    from bonk3.maps import load_map

    eng = BonkEngine(load_map("gang-grounds"))
    eng.reset([(0, 1), (1, 2)], seed=1234)
    out = eng.step([{"right": True}, {}])
    print(out.discs[0].x, out.discs[0].y, out.discs[0].energy)
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DYLIB = (REPO / "bonk-recreation/bonk-enviroment/rust/bonk-env/target/release"
         / "libbonk_env.dylib")

# Buffer layouts — must match rust/bonk-env/src/envcore.rs.
INPUT_STRIDE = 8    # present, up, down, left, right, action, action2, ml
DISC_STRIDE = 16
OUT_HEADER = 16
COLLISION_STRIDE = 9

# Disc field offsets within a DISC_STRIDE block.
D_PRESENT, D_X, D_Y, D_VX, D_VY = 0, 1, 2, 3, 4
D_ANGLE, D_ANGVEL, D_ENERGY, D_TEAM, D_ABILITY = 5, 6, 7, 8, 9
D_LAST_HIT_ID, D_LAST_HIT_TICKS, D_CHARGE, D_AIM, D_VTOL, D_SWING = 10, 11, 12, 13, 14, 15

# The heavy meter ("ability energy") runs 0..1000 exactly as in bonk.
ENERGY_MAX = 1000.0

DEFAULT_SETTINGS = {
    "mode": "b",          # b bonk | bs simple | v VTOL | sp grapple | ar arrows
    "engine": "b",
    "teamsEnabled": False,
    "winLimit": 3,
    "gameType": 0,
    "balance": {},
}


@dataclass
class Disc:
    """One player's physics state. x/y/vx/vy are in METRES."""
    present: bool
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    energy: float          # 0..1000, the heavy meter
    team: int
    ability_active: bool
    charge_state: float
    aim_angle: float

    @property
    def energy_frac(self) -> float:
        return self.energy / ENERGY_MAX


@dataclass
class StepResult:
    discs: list[Disc]
    deaths: list[tuple[int, int]]      # (disc_index, death_type) this frame
    round_frames: int
    freeze_time_end: float
    freeze_time_unfrozen: float
    game_complete: bool
    last_scorer: int

    @property
    def frozen(self) -> bool:
        """True during the round-start freeze, when inputs do not apply.
        Real bonk holds players for ~5 frames; our old sim never modelled it."""
        return self.round_frames < self.freeze_time_unfrozen


class BonkEngineError(RuntimeError):
    pass


def _load_lib(path: Path = DYLIB) -> ctypes.CDLL:
    if not path.exists():
        raise BonkEngineError(
            f"{path} not found — build it with python/bonk3/build.sh "
            "(needs the private bonk-io box2dweb-rs submodule)")
    lib = ctypes.CDLL(str(path))
    cd, cu8, csz = ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t
    vp, pd = ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
    lib.benv_new.restype, lib.benv_new.argtypes = vp, [cu8, csz, cu8, csz]
    lib.benv_free.argtypes = [vp]
    lib.benv_reset.restype = ctypes.c_int
    lib.benv_reset.argtypes = [vp, cu8, csz, cd, ctypes.c_bool, ctypes.c_bool]
    lib.benv_set_mode.argtypes = [vp, cu8, csz]
    lib.benv_step_into.restype = ctypes.c_int
    lib.benv_step_into.argtypes = [vp, pd, csz, cd, ctypes.c_uint, pd, csz]
    lib.benv_snapshot_into.restype = ctypes.c_int
    lib.benv_snapshot_into.argtypes = [vp, pd, csz]
    lib.benv_state_json.restype = ctypes.c_int64
    lib.benv_state_json.argtypes = [vp, ctypes.POINTER(ctypes.c_uint8), csz]
    lib.benv_set_state_json.restype = ctypes.c_int
    lib.benv_set_state_json.argtypes = [vp, cu8, csz]
    lib.benv_last_error.restype = csz
    lib.benv_last_error.argtypes = [vp, ctypes.POINTER(ctypes.c_uint8), csz]
    return lib


_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


class BonkEngine:
    """One bonk game instance. Not thread-safe; make one per env."""

    def __init__(self, map_data: dict, settings: dict | None = None,
                 max_players: int = 2, max_collisions: int = 64):
        self.lib = _lib()
        self.map_data = map_data
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.max_players = max_players

        map_b = json.dumps(map_data).encode()
        set_b = json.dumps(self.settings).encode()
        self._env = self.lib.benv_new(map_b, len(map_b), set_b, len(set_b))
        if not self._env:
            raise BonkEngineError("benv_new failed (see stderr for the parse error)")

        n = max_players
        self._out = np.zeros(
            OUT_HEADER + n * DISC_STRIDE + n * 2 + max_collisions * COLLISION_STRIDE,
            dtype=np.float64)
        self._inp = np.zeros(n * INPUT_STRIDE, dtype=np.float64)
        self._out_p = self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._inp_p = self._inp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # ── lifecycle ──────────────────────────────────────────────────────────
    def close(self):
        if getattr(self, "_env", None):
            self.lib.benv_free(self._env)
            self._env = None

    def __del__(self):
        self.close()

    def _err(self) -> str:
        buf = (ctypes.c_uint8 * 512)()
        n = self.lib.benv_last_error(self._env, buf, len(buf))
        return bytes(buf[:n]).decode("utf-8", "replace")

    def _check(self, rc: int, what: str):
        if rc != 0:
            raise BonkEngineError(f"{what}: {self._err()}")

    # ── control ────────────────────────────────────────────────────────────
    def set_mode(self, mode: str):
        b = mode.encode()
        self.lib.benv_set_mode(self._env, b, len(b))

    def reset(self, players: list[tuple[int, int]] | None = None, seed: float = 0.0,
              quick_start: bool = False, is_first_round: bool = True) -> StepResult:
        """players: list of (id, team). Team 1 = FFA/red in practice."""
        players = players or [(0, 1), (1, 2)][:self.max_players]
        pj = json.dumps([{"id": i, "team": t} for i, t in players]).encode()
        self._check(self.lib.benv_reset(self._env, pj, len(pj), float(seed),
                                        quick_start, is_first_round), "reset")
        return self.snapshot()

    def snapshot(self) -> StepResult:
        self._check(self.lib.benv_snapshot_into(self._env, self._out_p,
                                                len(self._out)), "snapshot")
        return self._decode()

    def step(self, inputs: list[dict | None], fps: float = 30.0,
             sub_steps: int = 1) -> StepResult:
        """inputs: one dict per player with any of
        up/down/left/right/action/action2/ml (action = heavy). None = absent."""
        self._inp[:] = 0.0
        for p, inp in enumerate(inputs[:self.max_players]):
            if inp is None:
                continue
            o = p * INPUT_STRIDE
            self._inp[o] = 1.0
            self._inp[o + 1] = float(bool(inp.get("up")))
            self._inp[o + 2] = float(bool(inp.get("down")))
            self._inp[o + 3] = float(bool(inp.get("left")))
            self._inp[o + 4] = float(bool(inp.get("right")))
            self._inp[o + 5] = float(bool(inp.get("action") or inp.get("heavy")))
            self._inp[o + 6] = float(bool(inp.get("action2")))
            self._inp[o + 7] = float(bool(inp.get("ml")))
        self._check(self.lib.benv_step_into(self._env, self._inp_p, len(self._inp),
                                            float(fps), int(sub_steps),
                                            self._out_p, len(self._out)), "step")
        return self._decode()

    def state_json(self) -> dict:
        """Full GameState as JSON — slow, for debugging/parity work only."""
        cap = 1 << 20
        while True:
            buf = (ctypes.c_uint8 * cap)()
            n = self.lib.benv_state_json(self._env, buf, cap)
            if n >= 0:
                return json.loads(bytes(buf[:n]))
            cap = -n + 1024        # negative = required capacity

    def set_state_json(self, state: dict) -> None:
        """Overwrite the full GameState. The only way to place discs anywhere
        other than a map spawn — `reset` picks spawns from team ids, so a spawn
        curriculum has to go through here."""
        b = json.dumps(state).encode()
        self._check(self.lib.benv_set_state_json(self._env, b, len(b)),
                    "set_state_json")

    # ── readout ────────────────────────────────────────────────────────────
    def _decode(self) -> StepResult:
        o = self._out
        # The engine sizes discs to the highest LIVE index, so the list shrinks
        # when trailing players die (a dead disc 0 is zeroed in place and keeps
        # its slot, but a dead last disc disappears). Pad back to max_players so
        # discs[p] is always player p.
        n_discs = min(int(o[0]), self.max_players)
        discs = []
        for i in range(n_discs):
            b = OUT_HEADER + i * DISC_STRIDE
            discs.append(Disc(
                present=o[b + D_PRESENT] != 0.0,
                x=o[b + D_X], y=o[b + D_Y], vx=o[b + D_VX], vy=o[b + D_VY],
                angle=o[b + D_ANGLE], angular_velocity=o[b + D_ANGVEL],
                energy=o[b + D_ENERGY], team=int(o[b + D_TEAM]),
                ability_active=o[b + D_ABILITY] != 0.0,
                charge_state=o[b + D_CHARGE], aim_angle=o[b + D_AIM]))
        while len(discs) < self.max_players:
            discs.append(Disc(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
                              False, 0.0, 0.0))
        n_deaths = int(o[8])
        doff = OUT_HEADER + n_discs * DISC_STRIDE
        deaths = [(int(o[doff + k * 2]), int(o[doff + k * 2 + 1]))
                  for k in range(n_deaths)]
        return StepResult(
            discs=discs, deaths=deaths, round_frames=int(o[4]),
            freeze_time_end=o[1], freeze_time_unfrozen=o[2],
            game_complete=o[5] != 0.0, last_scorer=int(o[3]))
