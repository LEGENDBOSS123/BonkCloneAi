"""Render real bonk.io maps with pygame.

Draws the actual decoded geometry — every shape composed through its fixture
and its body transform, in the map's own render order and colours — rather than
the approximate platform list `bonk/play.py` used. Shape kinds mirror the
game's codec: 'bx' box, 'ci' circle, 'po' polygon.

Coordinates: map geometry is in PIXELS, physics is in METRES. Everything here
converts to metres via physics.pixelsPerMeter, then to screen via Camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from .maps import play_center, ppm, to_world, world_bounds

BG = (5, 8, 10)
COL_PLAYER = [(0, 229, 255), (255, 82, 82), (120, 255, 140), (255, 214, 92)]
COL_DEATH = (255, 40, 40)
COL_SPAWN = (90, 220, 140)


def _rgb(color: int | None, fallback=(58, 90, 64)) -> tuple[int, int, int]:
    if color is None:
        return fallback
    return ((color >> 16) & 255, (color >> 8) & 255, color & 255)


def _rot(x: float, y: float, a: float) -> tuple[float, float]:
    if not a:
        return x, y
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


@dataclass
class Camera:
    """World metres -> screen pixels."""
    cx: float
    cy: float
    scale: float
    w: int
    h: int

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (int(self.w / 2 + (x - self.cx) * self.scale),
                int(self.h / 2 + (y - self.cy) * self.scale))

    # The browser renders a fixed world height regardless of map size; auto-
    # fitting to geometry instead makes the playfield vanish, because bonk maps
    # embed 999 m backdrops and ~200 m walls.
    WORLD_HEIGHT = 52.0

    @classmethod
    def fit(cls, map_data: dict, w: int, h: int,
            world_height: float | None = None) -> Camera:
        """Fixed-height view centred on the spawns, like the real game."""
        cx, cy = play_center(map_data)
        return cls(cx, cy, h / (world_height or cls.WORLD_HEIGHT), w, h)

    @classmethod
    def fit_geometry(cls, map_data: dict, w: int, h: int, margin: float = 1.12) -> Camera:
        """Zoom to the map's real (backdrop-filtered) extents instead."""
        x0, y0, x1, y1 = world_bounds(map_data)
        span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        scale = min(w / (span_x * margin), h / (span_y * margin))
        return cls((x0 + x1) / 2, (y0 + y1) / 2, scale, w, h)

    def follow(self, discs, lerp: float = 0.12):
        """Ease toward the midpoint of the live players."""
        pts = [(d.x, d.y) for d in discs if d.present]
        if not pts:
            return
        tx = sum(p[0] for p in pts) / len(pts)
        ty = sum(p[1] for p in pts) / len(pts)
        self.cx += (tx - self.cx) * lerp
        self.cy += (ty - self.cy) * lerp


class MapRenderer:
    """Precomputes each fixture's geometry in metres so drawing is cheap."""

    def __init__(self, map_data: dict):
        self.map = map_data
        ph = map_data["physics"]
        self.scale = ppm(map_data)
        self.shapes = ph["shapes"]
        self.fixtures = ph["fixtures"]
        self.bodies = ph["bodies"]
        # bodyRenderOrder is FRONT-to-back: the official renderer walks it in
        # reverse (bonk-renderer drawBodies: `for oi = len-1; oi >= 0; oi--`).
        # Walking it forwards paints the 999 m backdrop body last, over the
        # entire map.
        order = ph.get("bodyRenderOrder") or list(range(len(self.bodies)))
        self.order = list(reversed(order))

    def _shape_points(self, sh: dict) -> list[tuple[float, float]] | None:
        """Local-space outline in metres, or None for circles."""
        s = self.scale
        cx, cy = sh["center"][0] / s, sh["center"][1] / s
        t = sh["type"]
        if t == "bx":
            hw, hh = sh["width"] / (2 * s), sh["height"] / (2 * s)
            a = sh.get("angle", 0.0)
            return [(cx + dx, cy + dy) for dx, dy in
                    (_rot(sx * hw, sy * hh, a)
                     for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1)))]
        if t == "po":
            vs = sh.get("vertices") or []
            a = sh.get("angle", 0.0)
            pts = [_rot(v[0] / s, v[1] / s, a) for v in vs]
            return [(cx + px, cy + py) for px, py in pts] if len(pts) >= 3 else None
        return None   # 'ci' handled separately

    def draw(self, surf: pygame.Surface, cam: Camera, show_death=True,
             live_bodies: list | None = None):
        """`live_bodies` = engine state's physics.bodies (already in METRES,
        no origin offset). Pass it to animate moving platforms; omit it and
        the static map placement is used, which is correct for static maps."""
        for bi in self.order:
            if bi >= len(self.bodies):
                continue
            if live_bodies is not None and bi < len(live_bodies) and live_bodies[bi]:
                lb = live_bodies[bi]
                bx, by = lb["position"][0], lb["position"][1]
                ba = lb.get("angle", 0.0)
            else:
                body = self.bodies[bi]
                bx, by = to_world(body["position"][0], body["position"][1], self.scale)
                ba = body.get("angle", 0.0)
            body = self.bodies[bi]
            for fi in body["fixtureIndices"]:
                if fi >= len(self.fixtures):
                    continue
                fx = self.fixtures[fi]
                sh = self.shapes[fx["shapeIndex"]]
                col = COL_DEATH if (show_death and fx.get("death")) else _rgb(fx.get("color"))
                pts = self._shape_points(sh)
                if pts is None:                      # circle
                    cx, cy = sh["center"][0] / self.scale, sh["center"][1] / self.scale
                    wx, wy = _rot(cx, cy, ba)
                    p = cam.to_screen(bx + wx, by + wy)
                    r = max(1, int(sh["radius"] / self.scale * cam.scale))
                    pygame.draw.circle(surf, col, p, r)
                else:
                    scr = []
                    for px, py in pts:
                        wx, wy = _rot(px, py, ba)
                        scr.append(cam.to_screen(bx + wx, by + wy))
                    if len(scr) >= 3:
                        pygame.draw.polygon(surf, col, scr)

    def draw_spawns(self, surf: pygame.Surface, cam: Camera):
        for sp in self.map["spawns"]:
            p = cam.to_screen(*to_world(sp["x"], sp["y"], self.scale))
            pygame.draw.circle(surf, COL_SPAWN, p, 4, 1)


def draw_discs(surf: pygame.Surface, cam: Camera, discs, radius_m: float = 1.25):
    """Players. Ring thickness tracks the heavy meter, like the game's tint."""
    for i, d in enumerate(discs):
        if not d.present:
            continue
        col = COL_PLAYER[i % len(COL_PLAYER)]
        p = cam.to_screen(d.x, d.y)
        r = max(2, int(radius_m * cam.scale))
        pygame.draw.circle(surf, col, p, r)
        if d.ability_active:
            pygame.draw.circle(surf, (255, 255, 255), p, r,
                               max(1, int(r * 0.30 * d.energy_frac + 1)))
