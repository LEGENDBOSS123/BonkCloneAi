import { Map } from "../map/map.mjs";
import { Player } from "../entities/player.mjs";
import { Rectangle, Circle, Polygon } from "../platforms/index.mjs";

// Parses a Bonk.io map (the JSON the in-game editor exports) into one of our
// Map objects. The bonk schema (only the bits we use):
//
//   physics.shapes   : geometry, indexed by fixtures. type "bx" (box: w,h,a),
//                      "ci" (circle: r), "po" (polygon: v=verts, s=scale). Each
//                      has a local center `c` = [x, y].
//   physics.fixtures : appearance/material for a shape (`sh` index). `f` color,
//                      `re` restitution, `fr` friction, `np` "no physics".
//   physics.bodies   : groups of fixtures (`fx` indices) placed at `p` = [x, y]
//                      with angle `a`; `fric` is the body's default friction.
//   physics.ppm      : pixels per meter (bonk coords are pixels; +y is down,
//                      which already matches our world).
//   spawns           : player start positions.
//
// All bonk coordinates are scaled by 1/ppm so the result is in our meters.

const DEFAULT_PPM = 100;
// Bonk's ball radius in bonk pixels (approx); scaled like everything else so the
// player stays proportional to the map. Override via opts.playerRadius.
const BONK_BALL_RADIUS = 10;

// Fetch a bonk map JSON file and parse it. `url` may be a string or URL.
export async function loadBonkMap(url, opts = {}) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`bonkmap: failed to load ${url} (${res.status})`);
    return parseBonkMap(await res.json(), opts);
}

// Build a Map from an already-parsed bonk JSON object.
export async function parseBonkMap(data, { gravity = 20, scale, spawnName, playerRadius } = {}) {
    const physics = data.physics ?? {};
    const ppm = physics.ppm || DEFAULT_PPM;
    const s = scale ?? 1 / ppm;

    const map = await Map.create({
        name: data.m?.n ?? "Bonk Map",
        author: data.m?.a ?? "unknown",
        gravity,
    });
    map.ppm = ppm;
    const shapes = physics.shapes ?? [];
    const fixtures = physics.fixtures ?? [];
    const bodies = physics.bodies ?? [];

    // `bro` (body render order) lists body indices front-to-back, so we mount
    // them in reverse: the last entry (e.g. the background) is added first and
    // renders behind, the first entry is added last and renders on top. Fall
    // back to plain index order if `bro` is missing.
    const order = physics.bro?.length
        ? [...physics.bro].reverse()
        : bodies.map((_, i) => i);

    for (const bodyIndex of order) {
        const body = bodies[bodyIndex];
        if (!body) continue;
        const [bx, by] = body.p ?? [0, 0];
        const ba = body.a ?? 0;
        for (const fxIndex of body.fx ?? []) {
            const fixture = fixtures[fxIndex];
            if (!fixture) continue;
            const def = shapes[fixture.sh];
            if (!def) continue;
            const platform = shapeFromBonk(body, def, fixture, { bx, by, ba, bodyFric: body.fric, scale: s });
            if (platform) map.addShape(platform);
        }
    }

    // Two players share the map. The spawn points (in bonk pixels, scaled to
    // meters) are stored on the map so the env can re-randomize which player gets
    // which on every reset. The first player is the controllable one the camera
    // follows; the second just sits, affected only by physics.
    const [spawn1, spawn2] = pickSpawnPoints(data.spawns ?? [], spawnName);
    map.spawnPoints = [
        { x: spawn1.x * s, y: spawn1.y * s },
        { x: spawn2.x * s, y: spawn2.y * s },
    ];

    // Randomly assign the two spawn points so player 1 isn't always on the same one.
    const spawnOrder = Math.random() < 0.5 ? [0, 1] : [1, 0];
    map.addPlayer(
        new Player({ ...map.spawnPoints[spawnOrder[0]], radius: 1, color: 0x00e5ff, controllable: true })
    );
    map.addPlayer(
        new Player({ ...map.spawnPoints[spawnOrder[1]], radius: 1, color: 0xff5252, controllable: false })
    );

    return map;
}

// Rotate a local offset by the body's angle (radians).
function rotate(x, y, a) {
    if (!a) return { x, y };
    const cos = Math.cos(a);
    const sin = Math.sin(a);
    return { x: x * cos - y * sin, y: x * sin + y * cos };
}

// bonk `re` (restitution): null/negative means "default" (no bounce). Clamp to
// [0, 1] — bonk allows >1 super-bounce, but that destabilizes the solver.
function bounceOf(body, fixture) {
    let re = (fixture.re ?? body.re ?? -1);
    return re;
}

// bonk friction lives on the fixture (`fr`) and falls back to the body (`fric`).
function frictionOf(body, fixture, bodyFric) {
    const fr = (fixture.fr ?? body.fric);
    if(body.fricp == false){
        return 0;
    }
    return fr;
}

function shapeFromBonk(body, def, fixture, { bx, by, ba, bodyFric, scale }) {
    const common = {
        // bonk colors are already 0xRRGGBB ints; undefined falls back to default.
        color: fixture.f != null ? fixture.f : undefined,
        bounciness: bounceOf(body, fixture),
        friction: frictionOf(body, fixture, bodyFric),
        // bonk "no physics" fixtures are decorative: drawn but not collidable.
        solid: fixture.np !== true,
    };

    const [cx, cy] = def.c ?? [0, 0];
    const off = rotate(cx, cy, ba);
    const worldX = (bx + off.x) * scale;
    const worldY = (by + off.y) * scale;

    // Total rotation about the shape's center: the body's angle plus the
    // shape's own. Circles are rotation-invariant; boxes and polygons bake the
    // angle into their (convex) vertices, since our Rectangle/Polygon render
    // and collide in shape-local axis-aligned coordinates.
    const angle = (ba || 0) + (def.a || 0);

    switch (def.type) {
        case "bx": {
            const hw = (def.w * scale) / 2;
            const hh = (def.h * scale) / 2;
            // Axis-aligned: a plain Rectangle (origin is its top-left corner).
            if (!angle) {
                return new Rectangle({ ...common, x: worldX - hw, y: worldY - hh, width: hw * 2, height: hh * 2 });
            }
            // Rotated: emit the four rotated corners as a convex polygon.
            const corners = [
                { x: -hw, y: -hh },
                { x: hw, y: -hh },
                { x: hw, y: hh },
                { x: -hw, y: hh },
            ].map((p) => rotate(p.x, p.y, angle));
            return new Polygon({ ...common, x: worldX, y: worldY, points: corners });
        }
        case "ci":
            return new Circle({ ...common, x: worldX, y: worldY, radius: def.r * scale });
        case "po": {
            const polyScale = (def.s ?? 1) * scale;
            const points = (def.v ?? []).map(([px, py]) => rotate(px * polyScale, py * polyScale, angle));
            try {
                return new Polygon({ ...common, x: worldX, y: worldY, points });
            } catch (e) {
                console.warn(`bonkmap: skipping invalid polygon — ${e.message}`);
                return null;
            }
        }
        default:
            console.warn(`bonkmap: unknown shape type "${def.type}"`);
            return null;
    }
}

// Pick a spawn: by name if given, otherwise the highest-priority one.
function pickSpawn(spawns, name) {
    if (!spawns.length) return null;
    if (name) {
        const byName = spawns.find((sp) => sp.n === name);
        if (byName) return byName;
    }
    return spawns.reduce((best, sp) => (sp.priority > (best?.priority ?? -Infinity) ? sp : best), null);
}

// Distance (in bonk pixels) to nudge a second player apart when the map only
// provides a single usable spawn point.
const SECOND_SPAWN_OFFSET = 40;

// Return two spawn points (in bonk pixel coords) for the two players. Uses the
// two highest-priority spawns if the map has them; otherwise falls back to the
// primary spawn and an offset copy so the players don't overlap.
function pickSpawnPoints(spawns, name) {
    const primary = pickSpawn(spawns, name) ?? { x: 0, y: 0 };
    const p1 = { x: primary.x ?? 0, y: primary.y ?? 0 };

    // Prefer a distinct second spawn from the map (highest priority that isn't
    // the primary); otherwise offset the primary horizontally.
    const secondary = spawns.filter((sp) => sp !== primary)
        .reduce((best, sp) => (sp.priority > (best?.priority ?? -Infinity) ? sp : best), null);
    const p2 = secondary
        ? { x: secondary.x ?? 0, y: secondary.y ?? 0 }
        : { x: p1.x + SECOND_SPAWN_OFFSET, y: p1.y };

    return [p1, p2];
}
