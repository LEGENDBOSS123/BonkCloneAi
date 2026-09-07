# bonk3 — the real bonk.io engine, driven from Python

`bonk/sim.py` is a Box2D port of *our own JS clone*, verified to ~1e-5 against
that clone. `bonk-recreation/bonk-enviroment` is a **bit-identical**
reimplementation of the actual game, and it ships a Rust core with a plain C ABI
— so Python can drive the real physics directly, with no JS runtime and no
per-frame serialization.

This package is that bridge, plus accurate map parsing and rendering.

| file | what it is |
|---|---|
| `engine.py` | ctypes binding to `libbonk_env.dylib` (`benv_*`). Flat f64 buffers in/out. |
| `maps.py` | load decoded `MapData`, coordinate helpers, spawns, bounds |
| `render.py` | pygame renderer for real bonk geometry (bx/ci/po, body transforms, map colours) |
| `view.py` | interactive viewer / headless screenshotter |
| `tools/decode_maps.mjs` | node: bonk database strings → `MapData` JSON (offline, one-time) |
| `build.sh` | fetch the private submodule + `cargo build --release` + smoke test |

## Setup

```bash
python/bonk3/build.sh                                              # build the engine
node python/bonk3/tools/decode_maps.mjs --name "gang grounds 2.0"  # decode a map
python -m bonk3.view --map gang-grounds-2-0                        # look at it
```

`build.sh` needs access to the private `bonk-io/bonk2-box2d-rs` submodule.

## Using your own map

Training/eval/play all read `bonk2/config.py`'s `MAP_NAME`, which takes either a
**decoded-map stem** or a **path to any MapData JSON**:

```python
MAP_NAME = "gang-grounds-2-0"        # stem in python/bonk3/maps/
MAP_NAME = "/abs/path/to/mine.json"  # or any decoded MapData file
```

Three ways to get a map into that form — all keep the same engine:

```bash
# 1. YOUR map, from the share code bonk gives you
node python/bonk3/tools/decode_maps.mjs --string "ILAcCFgRWBhKDGBz..."
node python/bonk3/tools/decode_maps.mjs --code-file mymap.txt --out /tmp/mine.json

# 2. Any of the 2137 bundled maps, by name or index
node python/bonk3/tools/decode_maps.mjs --list --max 40      # browse, simplest first
node python/bonk3/tools/decode_maps.mjs --name "bounce ball"
node python/bonk3/tools/decode_maps.mjs --index 322

# 3. Bulk, for multi-map work later
node python/bonk3/tools/decode_maps.mjs --all --max 50
```

Each writes `python/bonk3/maps/<slug>.json`; set `MAP_NAME` to that stem.
Check it before committing a run — complex maps cost far more per step:

```bash
python -m bonk3.view --map <slug>          # look at it
python -m bonk3.view --map <slug> --shot /tmp/m.png --frames 5   # headless
```

Note the input must be the game's **map code** (a long base64-ish blob), not a
URL and not a compact map JSON like `src/bonkmap/map1.json` — that older format
needs `encodeToDatabase`, which lives in the `bonk-map` package rather than the
bundled decoder.

## Using the engine

```python
from bonk3.engine import BonkEngine
from bonk3.maps import load_map

eng = BonkEngine(load_map("gang-grounds-2-0"))
state = eng.reset([(0, 1), (1, 2)], seed=1234)
state = eng.step([{"right": True}, {"heavy": True}])
d = state.discs[0]
print(d.x, d.y, d.vx, d.vy, d.energy)   # metres, and heavy 0..1000
```

`StepResult` carries `discs`, `deaths` (this frame), `round_frames`,
`game_complete`, and `frozen`. Modes via `settings={"mode": "ar"}` or
`eng.set_mode(...)`: `b` bonk, `bs` simple, `v` VTOL, `sp` grapple, `ar` arrows.

## Things that bit me — read before extending

**Map coordinates are re-origined.** The engine does
`world = (map_pixels + MAP_HALF) / ppm` with `MAP_HALF = (365, 250)`, applied to
**body positions and spawns** but *not* shape centers (those are body-local and
only divided by `ppm`). Skip the offset and geometry renders in the right shape
at the wrong place. Same 365/250 constants `play3.mjs` uses.

**`bodyRenderOrder` is front-to-back.** The official renderer walks it in
*reverse*. Walking forwards paints the 999 m backdrop body last, over everything.

**Maps embed enormous shapes.** 999 m backdrop slabs and ~200 m walls are
normal, so an auto-fit camera zooms out until the playfield is invisible. The
game renders a fixed ~52 m view height instead — that's `Camera.fit`;
`Camera.fit_geometry` is the (filtered) zoom-to-extents alternative.

**`ppm` is clamped** to [2, 300] with a fallback of 15, engine-side. `maps.ppm()`
replicates that.

**`discs` is index-stable but variable-length.** A dead disc 0 is zeroed in
place and keeps its slot; a dead *trailing* disc disappears, so the engine's
list can be shorter than the player count. `engine.py` pads back to
`max_players`, so `state.discs[p]` is always player p — check `.present`.

**Rounds start frozen** for ~5 frames (`state.frozen`). Our old sim never
modelled this; inputs during the freeze do nothing.

## Performance

Measured, 320 envs in one process, 2 players:

| engine | map | env-steps/s |
|---|---|---|
| `bonk/sim.py` (Python + Box2D) | Gang Grounds (22 shapes) | 183,707 |
| **bonk3 (Rust, bit-identical)** | same map | **198,495** |
| bonk3 | food fight (83 shapes) | 11,642 |

So the real engine is **not slower** than what we train on today — at equal map
complexity it is slightly faster. The food-fight number is map cost, not engine
overhead: complex maps are ~17x more expensive, which matters if multi-map
training ever happens.

Determinism is bit-exact: same seed + same inputs reproduce identical f64s, so
replays and rollback resimulation work.

## What this does NOT do yet

`bonk3` is the engine/map/render layer. **It is not wired into training** —
`bonk2` still runs on `bonk/sim.py`. Switching the trainer over means a new
observation built from `Disc` fields (`energy` replaces our heavy tracking,
plus angle/angular velocity that we never had), which changes `STATE_DIM` and
therefore means retraining from scratch and updating `play3.mjs`. That is a
deliberate, separate decision — see the notes in the parent `CLAUDE.md` about
obs changes.

Moving platforms: `render.draw(..., live_bodies=...)` accepts the engine
state's `physics.bodies` for animation; without it, static map placement is
used, which is correct for static maps.
