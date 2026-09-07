"""Load real bonk.io maps (MapData JSON produced by tools/decode_maps.mjs).

Map geometry is stored in PIXELS; divide by `physics.pixelsPerMeter` to get the
metres the engine and the observation use. Helpers here do that for you.
"""

from __future__ import annotations

import json
from pathlib import Path

MAPS_DIR = Path(__file__).resolve().parent / "maps"

# The engine re-origins the map before simulating (rust generic.rs):
#     world = (map_pixels + MAP_HALF) / ppm
# applied to BODY positions and spawns; shape centers stay body-local and are
# only divided by ppm. These are the same 365/250 constants play3.mjs uses to
# convert the live game's screen coords back to world coords. Get this wrong
# and geometry renders in the right shape at the wrong place.
MAP_HALF_W = 730.0 / 2.0
MAP_HALF_H = 500.0 / 2.0

# The engine clamps an out-of-range ppm rather than trusting the map.
PPM_MIN, PPM_MAX, PPM_FALLBACK = 2.0, 300.0, 15.0


def available() -> list[str]:
    return sorted(p.stem for p in MAPS_DIR.glob("*.json"))


def load_map(name_or_path: str | Path) -> dict:
    """Load by stem ("gang-grounds-2-0") or explicit path."""
    p = Path(name_or_path)
    if not p.exists():
        p = MAPS_DIR / f"{Path(name_or_path).stem}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{name_or_path} not found. Have: {available() or '(none — run '
            'node python/bonk3/tools/decode_maps.mjs --name \"...\")'}")
    with open(p) as f:
        return json.load(f)


def ppm(map_data: dict) -> float:
    """Effective pixels-per-metre, clamped exactly as the engine clamps it."""
    p = float(map_data["physics"]["pixelsPerMeter"])
    return p if PPM_MIN <= p <= PPM_MAX else PPM_FALLBACK


def to_world(x_px: float, y_px: float, scale: float) -> tuple[float, float]:
    """Map pixels -> engine world metres (body/spawn placement)."""
    return (x_px + MAP_HALF_W) / scale, (y_px + MAP_HALF_H) / scale


def summary(map_data: dict) -> dict:
    md, ph = map_data["metadata"], map_data["physics"]
    return {
        "name": md["name"], "author": md["author"], "dbid": md["databaseId"],
        "shapes": len(ph["shapes"]), "bodies": len(ph["bodies"]),
        "joints": len(ph["joints"]), "spawns": len(map_data["spawns"]),
        "ppm": ph["pixelsPerMeter"],
    }


def spawns(map_data: dict, meters: bool = True) -> list[dict]:
    """Spawn points, highest priority first (the game picks in that order)."""
    s = ppm(map_data) if meters else 1.0
    out = []
    for sp in map_data["spawns"]:
        wx, wy = to_world(sp["x"], sp["y"], s) if meters else (sp["x"], sp["y"])
        out.append({**sp, "x": wx, "y": wy})
    return sorted(out, key=lambda sp: -sp.get("priority", 0))


def world_bounds(map_data: dict, meters: bool = True,
                 max_extent: float = 120.0) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) in metres over shapes placed through their
    BODY transform (shape centers are body-local, so ignoring the body position
    puts geometry in the wrong place).

    Shapes bigger than `max_extent` metres are skipped: bonk maps routinely use
    999 m backdrop slabs and ~200 m walls, and including them makes any auto-fit
    camera zoom out until the playfield is a few pixels wide.
    """
    s = ppm(map_data) if meters else 1.0
    ph = map_data["physics"]
    shapes, fixtures = ph["shapes"], ph["fixtures"]
    xs, ys = [], []
    for body in ph["bodies"]:
        bx, by = to_world(body["position"][0], body["position"][1], s)
        for fi in body["fixtureIndices"]:
            if fi >= len(fixtures):
                continue
            sh = shapes[fixtures[fi]["shapeIndex"]]
            cx, cy = sh["center"][0] / s, sh["center"][1] / s
            if sh["type"] == "bx":
                hw, hh = sh["width"] / (2 * s), sh["height"] / (2 * s)
            elif sh["type"] == "ci":
                hw = hh = sh["radius"] / s
            else:
                vs = sh.get("vertices") or []
                if not vs:
                    continue
                hw = max(abs(v[0]) for v in vs) / s
                hh = max(abs(v[1]) for v in vs) / s
            if hw * 2 > max_extent or hh * 2 > max_extent:
                continue
            xs += [bx + cx - hw, bx + cx + hw]
            ys += [by + cy - hh, by + cy + hh]
    if not xs:
        return (-30.0, -30.0, 30.0, 30.0)
    return min(xs), min(ys), max(xs), max(ys)


def play_center(map_data: dict) -> tuple[float, float]:
    """Centroid of the spawn points in metres — where the action actually is."""
    sp = map_data["spawns"]
    if not sp:
        x0, y0, x1, y1 = world_bounds(map_data)
        return (x0 + x1) / 2, (y0 + y1) / 2
    s = ppm(map_data)
    pts = [to_world(p["x"], p["y"], s) for p in sp]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
