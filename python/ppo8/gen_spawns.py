"""Generate a pool of SURVIVABLE spawn positions for a map.

The `death` map is two small spawn platforms separated by a lethal pit (plus a
lower centre slab). A disc dropped at a random point either lands on real ground
or falls into the pit and dies-then-respawns. `present`-at-end is NOT a survival
test — a respawn leaves the disc present. We detect a DEATH EVENT inside a 10 s
idle window instead, and record where survivors SETTLE (guaranteed stable ground).

Training then samples two disc positions from this pool, so (a) neither disc ever
starts in a death trap it cannot escape, and (b) "self" appears across the whole
map rather than only at seat 0's spawn — which is what makes both seats and the
mirror-augmentation valid.

    python -m ppo6.gen_spawns --n 6000 --out ppo6/spawns_death.json
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from . import config as C
from bonk3.simadapter import make_sim

IDLE = [None, None]


def survives(sim, x, y, frames=300):
    """Place disc 0 at (x,y) at rest, idle `frames` ticks. Return the SETTLED
    (x,y) if no death event fires for disc 0, else None."""
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
    sim.engine.set_state_json(st)
    res = None
    for _ in range(frames):
        res = sim.engine.step(IDLE)
        if (not res.discs[0].present) or any(idx == 0 for idx, _ in res.deaths):
            return None
    p = res.discs[0]
    # A disc still moving fast at the end hasn't settled — reject (edge cases
    # that are sliding toward a pit).
    if abs(getattr(p, "vx", 0.0)) + abs(getattr(p, "vy", 0.0)) > 0.5:
        return None
    return (float(p.x), float(p.y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="candidate samples")
    ap.add_argument("--frames", type=int, default=300, help="idle ticks (~10s)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  f"spawns_{C.MAP_NAME}.json"))
    ap.add_argument("--xlo", type=float, default=0.0)
    ap.add_argument("--xhi", type=float, default=73.0)
    ap.add_argument("--ylo", type=float, default=5.0)
    ap.add_argument("--yhi", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    sim = make_sim(C.MAP_NAME, engine=C.ENGINE)

    survivors = []
    for i in range(args.n):
        x = random.uniform(args.xlo, args.xhi)
        y = random.uniform(args.ylo, args.yhi)
        pos = survives(sim, x, y, args.frames)
        if pos is not None:
            survivors.append(pos)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{args.n} sampled, {len(survivors)} survivable")

    # Report the platform structure: round settled positions to a coarse grid
    # and count — reveals the handful of real ground clusters.
    from collections import Counter
    clusters = Counter((round(x / 3) * 3, round(y / 3) * 3)
                       for x, y in survivors)
    print(f"\n  {len(survivors)} survivable positions on "
          f"{len(clusters)} coarse clusters:")
    for (cx, cy), n in clusters.most_common(12):
        print(f"    ~({cx:4.0f},{cy:4.0f})  x{n}")

    with open(args.out, "w") as f:
        json.dump({"map": C.MAP_NAME, "positions": survivors}, f)
    print(f"\n  wrote {len(survivors)} positions -> {args.out}")


if __name__ == "__main__":
    main()
