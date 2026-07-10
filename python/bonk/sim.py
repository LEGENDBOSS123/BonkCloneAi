"""Python port of the JS bonk-clone simulation.

Faithful translation of:
  src/bonkmap/bonkmap_parser.mjs (map JSON -> shapes/spawns, 1/ppm scaling)
  src/physics.mjs               (Box2D world, restitution mixing, ball body)
  src/entities/player.mjs       (movement forces, heavy mass, jump + ground ray)
  src/map/map.mjs               (fixed 30 TPS stepping)

planck.js is a JS port of Box2D 2.3, and pybox2d wraps the same engine, so the
solver semantics match (float32 in C vs float64 in JS causes gradual trajectory
divergence in chaotic contact sequences — expected and harmless).

Coordinates: meters, +y DOWN (gravity is +y), matching the JS world exactly.
"""

import math
import random

from Box2D import (  # type: ignore
    b2CircleShape,
    b2ContactListener,
    b2MassData,
    b2PolygonShape,
    b2RayCastCallback,
    b2Vec2,
    b2World,
)

from . import config as C


def mix_restitution(r1: float, r2: float) -> float:
    """src/physics.mjs: both positive -> max, else sum clamped to >= 0."""
    if r1 > 0 and r2 > 0:
        return max(r1, r2)
    return max(0.0, r1 + r2)


class _RestitutionMixer(b2ContactListener):
    def PreSolve(self, contact, old_manifold):  # noqa: N802 (Box2D API name)
        contact.restitution = mix_restitution(
            contact.fixtureA.restitution, contact.fixtureB.restitution
        )


class _GroundRay(b2RayCastCallback):
    def __init__(self, exclude_fixture):
        super().__init__()
        self.exclude = exclude_fixture
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):  # noqa: N802
        if fixture == self.exclude:
            return -1  # skip self, keep searching
        self.hit = True
        return fraction  # clip to nearest (existence is all we need)


def _rotate(x, y, a):
    if not a:
        return x, y
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


class Player:
    """Dynamic ball + the exact control law of src/entities/player.mjs."""

    def __init__(self, world: b2World, x: float, y: float):
        self.world = world
        self.radius = C.PLAYER_RADIUS
        self.heavy_power = C.HEAVY_MAX
        self.current_input = None

        self.body = world.CreateDynamicBody(
            position=(x, y), bullet=True, fixedRotation=True, linearDamping=0.0
        )
        self.fixture = self.body.CreateFixture(
            shape=b2CircleShape(radius=self.radius),
            density=1.0,
            friction=C.BALL_FRICTION,
            restitution=C.BALL_RESTITUTION,
        )
        self._set_mass(1.0)

    def _set_mass(self, mass: float):
        # physics.mjs setAdditionalMass: pin mass, centered COM, zero inertia.
        self.body.massData = b2MassData(mass=mass, center=(0.0, 0.0), I=0.0)

    # keys: dict with up/down/left/right/heavy booleans
    def control(self, keys):
        self.current_input = keys
        dir_x = (-1 if keys["left"] else 0) + (1 if keys["right"] else 0)
        dir_y = (1 if keys["down"] else 0) + (-1 if keys["up"] else 0)

        if keys["heavy"]:
            self.heavy_power = max(0.0, self.heavy_power - C.HEAVY_DRAIN)
        else:
            self.heavy_power = min(C.HEAVY_MAX, self.heavy_power + C.HEAVY_REGEN)

        if keys["heavy"]:
            self._set_mass(1.0 + C.HEAVY_EXTRA_MASS * self.heavy_power / C.HEAVY_MAX)
        else:
            self._set_mass(1.0)

        if dir_x or dir_y:
            self.body.ApplyForceToCenter(
                (dir_x * C.MOVE_ACCEL, dir_y * C.MOVE_ACCEL), True
            )

        if keys["up"] and self.is_grounded():
            self.jump()

    def is_grounded(self) -> bool:
        p = self.body.position
        cb = _GroundRay(self.fixture)
        reach = self.radius + C.GROUND_MARGIN
        self.world.RayCast(cb, (p.x, p.y), (p.x, p.y + reach))
        return cb.hit

    def jump(self):
        if abs(self.body.linearVelocity.y) >= C.JUMP_VY_THRESHOLD:
            return
        self.body.ApplyLinearImpulse(
            (0.0, -C.JUMP_SPEED * self.body.mass), self.body.worldCenter, True
        )

    # --- convenience accessors matching the JS PhysicsBody interface --------
    @property
    def pos(self):
        p = self.body.position
        return p.x, p.y

    @property
    def vel(self):
        v = self.body.linearVelocity
        return v.x, v.y

    def teleport(self, x, y):
        self.body.position = (x, y)
        self.body.linearVelocity = (0.0, 0.0)
        self.body.angularVelocity = 0.0
        self.body.awake = True


# --- bonk map parsing ---------------------------------------------------------
def _bounce_of(body, fixture):
    re = fixture.get("re")
    if re is None:
        re = body.get("re")
    return -1.0 if re is None else float(re)


def _friction_of(body, fixture):
    fr = fixture.get("fr")
    if fr is None:
        fr = body.get("fric")
    # JS: `body.fricp == false` (loose) -> only an explicit false/0 zeroes it.
    if body.get("fricp") in (False, 0):
        return 0.0
    return fr  # may be None -> Box2D default 0.2 (planck does the same)


def _shape_from_bonk(body, sdef, fixture, bx, by, ba, scale):
    common = {
        "color": fixture.get("f"),
        "bounciness": _bounce_of(body, fixture),
        "friction": _friction_of(body, fixture),
        "solid": fixture.get("np") is not True,
    }
    cx, cy = sdef.get("c", [0, 0])
    ox, oy = _rotate(cx, cy, ba)
    wx, wy = (bx + ox) * scale, (by + oy) * scale
    angle = (ba or 0) + (sdef.get("a") or 0)

    kind = sdef.get("type")
    if kind == "bx":
        hw = sdef["w"] * scale / 2
        hh = sdef["h"] * scale / 2
        if not angle:
            return {**common, "kind": "box", "x": wx - hw, "y": wy - hh,
                    "w": hw * 2, "h": hh * 2}
        pts = [_rotate(px, py, angle)
               for px, py in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
        return {**common, "kind": "polygon", "x": wx, "y": wy, "points": pts}
    if kind == "ci":
        return {**common, "kind": "circle", "x": wx, "y": wy,
                "radius": sdef["r"] * scale}
    if kind == "po":
        ps = (sdef.get("s") or 1) * scale
        pts = [_rotate(px * ps, py * ps, angle) for px, py in sdef.get("v", [])]
        if len(pts) < 3:
            return None
        return {**common, "kind": "polygon", "x": wx, "y": wy, "points": pts}
    return None


def _pick_spawn_points(spawns):
    if not spawns:
        return ({"x": 0, "y": 0}, {"x": 40, "y": 0})
    primary = max(spawns, key=lambda sp: sp.get("priority", float("-inf")))
    rest = [sp for sp in spawns if sp is not primary]
    secondary = max(rest, key=lambda sp: sp.get("priority", float("-inf"))) if rest else None
    p1 = {"x": primary.get("x", 0), "y": primary.get("y", 0)}
    p2 = ({"x": secondary.get("x", 0), "y": secondary.get("y", 0)} if secondary
          else {"x": p1["x"] + 40, "y": p1["y"]})
    return p1, p2


class BonkSim:
    """The two-player world: platforms + 2 balls, stepped at a fixed 30 TPS."""

    def __init__(self, map_json: dict, gravity: float = C.GRAVITY):
        physics = map_json.get("physics", {})
        self.ppm = physics.get("ppm", 100)
        scale = 1.0 / self.ppm

        self.world = b2World(gravity=(0.0, gravity))
        self._mixer = _RestitutionMixer()
        self.world.contactListener = self._mixer

        shapes = physics.get("shapes", [])
        fixtures = physics.get("fixtures", [])
        bodies = physics.get("bodies", [])
        order = list(reversed(physics.get("bro", []))) or range(len(bodies))

        self.platforms = []  # kept for rendering
        for bi in order:
            if bi >= len(bodies):
                continue
            body = bodies[bi]
            bx, by = body.get("p", [0, 0])
            ba = body.get("a", 0)
            for fxi in body.get("fx", []):
                if fxi >= len(fixtures):
                    continue
                fixture = fixtures[fxi]
                sdef = shapes[fixture["sh"]] if fixture.get("sh", -1) < len(shapes) else None
                if not sdef:
                    continue
                plat = _shape_from_bonk(body, sdef, fixture, bx, by, ba, scale)
                if plat:
                    self.platforms.append(plat)
                    if plat["solid"]:
                        self._add_platform_body(plat)

        s1, s2 = _pick_spawn_points(map_json.get("spawns", []))
        self.spawn_points = [
            (s1["x"] * scale, s1["y"] * scale),
            (s2["x"] * scale, s2["y"] * scale),
        ]
        self.players = [
            Player(self.world, *self.spawn_points[0]),
            Player(self.world, *self.spawn_points[1]),
        ]

        self.death_y = C.KILL_LINE_Y / self.ppm
        self.death_r2 = (C.KILL_RADIUS / self.ppm) ** 2
        self.tick_count = 0

    def _add_platform_body(self, plat):
        body = self.world.CreateStaticBody(position=(plat["x"], plat["y"]))
        kw = {}
        if plat["friction"] is not None:
            kw["friction"] = float(plat["friction"])
        kw["restitution"] = float(plat["bounciness"])
        try:
            if plat["kind"] == "box":
                hx, hy = plat["w"] / 2, plat["h"] / 2
                shape = b2PolygonShape()
                shape.SetAsBox(hx, hy, (hx, hy), 0.0)  # top-left origin, like JS
            elif plat["kind"] == "circle":
                shape = b2CircleShape(radius=plat["radius"])
            else:
                shape = b2PolygonShape(vertices=[(p[0], p[1]) for p in plat["points"]])
            body.CreateFixture(shape=shape, **kw)
        except Exception as e:  # >8 verts / degenerate — JS skips these too
            print(f"sim: skipping collider — {e}")

    def step(self, keys0: dict, keys1: dict):
        """One fixed tick: control both players, then step the world."""
        self.players[0].control(keys0)
        self.players[1].control(keys1)
        self.world.Step(C.DT, C.VELOCITY_ITERATIONS, C.POSITION_ITERATIONS)
        self.tick_count += 1

    def is_dead(self, i: int) -> bool:
        x, y = self.players[i].pos
        return y > self.death_y or (x * x + y * y) > self.death_r2

    def reset(self, randomize_spawns: bool = True):
        order = [0, 1] if (not randomize_spawns or random.random() < 0.5) else [1, 0]
        for i, player in enumerate(self.players):
            sx, sy = self.spawn_points[order[i]]
            player.teleport(sx, sy)
            player._set_mass(1.0)
            player.heavy_power = C.HEAVY_MAX
            player.current_input = None
        self.tick_count = 0


IDLE_KEYS = {"up": False, "down": False, "left": False, "right": False, "heavy": False}


def index_to_keys(index: int) -> dict:
    """Joint action index -> keys (src/rl2/lag_env.mjs indexToKeys)."""
    lr = index // 6
    ud = (index // 2) % 3
    return {
        "left": lr == 1,
        "right": lr == 2,
        "up": ud == 1,
        "down": ud == 2,
        "heavy": index % 2 == 1,
    }
