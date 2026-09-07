"""Generate the pool of SURVIVABLE spawn positions for a map.

A disc dropped at a random point either settles on real ground or falls into a
pit. `present`-at-end is NOT a survival test — with `respawnOnDeath` a dead disc
is present again a moment later — so we watch for a DEATH EVENT during an idle
window and record where survivors SETTLE, which is guaranteed stable ground.

Training samples two disc positions from this pool, so neither disc ever starts
in a trap it cannot escape, and "self" appears across the WHOLE map rather than
only at seat 0's spawn — which is what makes both seats, and the mirror
augmentation, valid.

Two things this fixes versus `ppo7/gen_spawns.py`, both of which silently cost
map coverage:

* **Bounds are derived from the map, not hardcoded.** ppo7 defaulted to
  `x in [0, 73], y in [5, 45]`, which are `gang-grounds-2-0`'s numbers
  (`2 * sim.origin`). Pointed at a map with a different extent they either
  clip it or waste every sample outside it, with no error either way. Here the
  probe box is `[0, 2*origin]` per axis, so it is right for whatever map is
  passed and can still be overridden.
* **The other disc is moved out of the way during the probe.** ppo7 left disc 1
  sitting at its map spawn, so any candidate near it was judged while being
  shoved by a disc that will not be there at training time. Measured on
  `gang-grounds-2-0`, that alone falsely rejects most of the x≈55-62 platform.
  It is parked on the map spawn FARTHEST from the candidate rather than
  off-map: dumping it into the void kills it, which ends the round and
  teleports the probe disc back to its own spawn, so every sample then
  "survives" at seat 0's spawn point and the pool collapses to a single spot.
  A teleport check below catches that class of failure rather than trusting it
  not to happen.

Positions are recorded where the disc SETTLES, so many (x, y) drops collapse
onto the same ground point; a platform's share of the pool ends up roughly
proportional to its width, which is what "spawn anywhere survivable" should
mean.

    python -m bonkenv.gen_spawns --map gang-grounds-2-0 --n 20000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import numpy as np

from bonk3.simadapter import make_sim

IDLE = [None, None]
MAX_DRIFT_M = 8.0        # settle further than this from the drop == it teleported


def map_spawn_points(sim) -> list[tuple[float, float]]:
    """The map's own declared spawn points, in engine metres."""
    ox, oy = sim.origin
    ppm = sim.ppm
    return [(s["x"] / ppm + ox, s["y"] / ppm + oy)
            for s in sim.map_data.get("spawns", [])]


def survives(sim, x: float, y: float, parks: list[tuple[float, float]],
             frames: int = 300, settle_speed: float = 0.5
             ) -> tuple[float, float] | None:
    """Place disc 0 at rest at `(x, y)`, idle `frames` ticks, and return the
    SETTLED position if no death event fired for it — else None.

    The other disc is moved to whichever of `parks` is farthest from the
    candidate: far enough not to shove it, but still on real ground so it stays
    alive and the round does not end under us.
    """
    sim.engine.reset(sim._teams(), seed=sim._seed)
    st = sim.engine.state_json()
    d = st.get("discs") or []
    if len(d) < 2:
        return None
    d[0]["x"], d[0]["y"] = x, y
    for k in ("velocityX", "velocityY"):
        if k in d[0]:
            d[0][k] = 0.0
    if "spawnX" in d[0]:
        d[0]["spawnX"], d[0]["spawnY"] = x, y
    if parks:
        px, py = max(parks, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        d[1]["x"], d[1]["y"] = px, py
        for k in ("velocityX", "velocityY"):
            if k in d[1]:
                d[1][k] = 0.0
    sim.engine.set_state_json(st)

    res = None
    for _ in range(frames):
        res = sim.engine.step(IDLE)
        if (not res.discs[0].present) or any(idx == 0 for idx, _ in res.deaths):
            return None
    p = res.discs[0]
    # Still moving at the end means it has not settled — typically sliding
    # toward a pit it would reach given a few more ticks.
    if abs(getattr(p, "vx", 0.0)) + abs(getattr(p, "vy", 0.0)) > settle_speed:
        return None
    # A disc falls essentially straight down, so a large horizontal jump means
    # the round reset and put it back on its spawn — recording that would fill
    # the pool with one point (and it did, before this check existed).
    if abs(float(p.x) - x) > MAX_DRIFT_M:
        return None
    return (float(p.x), float(p.y))


def stratified_samples(n: int, xlo: float, xhi: float, ylo: float, yhi: float,
                       rng: random.Random) -> list[tuple[float, float]]:
    """Jittered-grid samples over the box. Plain uniform sampling leaves
    thin platforms to luck; stratifying guarantees every column of the map is
    probed in proportion to its area, which is the whole point here."""
    cols = max(1, int(round((n * (xhi - xlo) / max(1e-9, yhi - ylo)) ** 0.5)))
    rows = max(1, int(round(n / cols)))
    dx, dy = (xhi - xlo) / cols, (yhi - ylo) / rows
    pts = [(xlo + (i + rng.random()) * dx, ylo + (j + rng.random()) * dy)
           for i in range(cols) for j in range(rows)]
    rng.shuffle(pts)
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", default="gang-grounds-2-0")
    ap.add_argument("--n", type=int, default=20000, help="candidate samples")
    ap.add_argument("--frames", type=int, default=300, help="idle ticks (~10 s)")
    ap.add_argument("--out", default=None,
                    help="default: bonkenv/spawns/spawns_<map>.json")
    # All four default to the map's own extent; pass them only to narrow it.
    ap.add_argument("--xlo", type=float, default=None)
    ap.add_argument("--xhi", type=float, default=None)
    ap.add_argument("--ylo", type=float, default=None)
    ap.add_argument("--yhi", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="probe and report coverage, write nothing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    sim = make_sim(args.map)

    # `origin` is the map centre in engine metres, so the playable box is
    # [0, 2*origin] on each axis — the map's own extent, whatever map this is.
    ox, oy = sim.origin
    xlo = args.xlo if args.xlo is not None else 0.0
    xhi = args.xhi if args.xhi is not None else 2.0 * ox
    ylo = args.ylo if args.ylo is not None else 0.0
    yhi = args.yhi if args.yhi is not None else 2.0 * oy
    print(f"  map {args.map}: probing x[{xlo:.1f},{xhi:.1f}] y[{ylo:.1f},{yhi:.1f}] "
          f"with {args.n} stratified samples, {args.frames} idle ticks")

    parks = map_spawn_points(sim)
    print(f"  parking the idle disc on the farthest of {len(parks)} map spawns")
    survivors: list[tuple[float, float]] = []
    pts = stratified_samples(args.n, xlo, xhi, ylo, yhi, rng)
    for i, (x, y) in enumerate(pts):
        pos = survives(sim, x, y, parks, args.frames)
        if pos is not None:
            survivors.append(pos)
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(pts)} sampled, {len(survivors)} survivable")

    if not survivors:
        raise SystemExit("no survivable positions found — check the bounds/map")

    arr = np.asarray(survivors, dtype=np.float64)
    print(f"\n  {len(survivors)} survivable positions"
          f"   x[{arr[:,0].min():.1f},{arr[:,0].max():.1f}]"
          f"   y[{arr[:,1].min():.1f},{arr[:,1].max():.1f}]")

    # Coverage across the map's width is the thing that silently regresses, so
    # print it: an empty bin is either a real pit or a pool that missed a side.
    hist, edges = np.histogram(arr[:, 0], bins=12, range=(xlo, xhi))
    print("  x coverage (12 bins across the map):")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        bar = "#" * int(40 * c / max(1, hist.max()))
        print(f"    [{lo:5.1f},{hi:5.1f}) {c:6d} {bar}")

    clusters = Counter((round(x / 3) * 3, round(y / 3) * 3) for x, y in survivors)
    print(f"\n  {len(clusters)} coarse ground clusters, top 12:")
    for (cx, cy), n in clusters.most_common(12):
        print(f"    ~({cx:4.0f},{cy:4.0f})  x{n}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return
    out = args.out or os.path.join(os.path.dirname(__file__), "spawns",
                                   f"spawns_{args.map}.json")
    with open(out, "w") as f:
        json.dump({"map": args.map, "positions": survivors}, f)
    print(f"\n  wrote {len(survivors)} positions -> {out}")


if __name__ == "__main__":
    main()
