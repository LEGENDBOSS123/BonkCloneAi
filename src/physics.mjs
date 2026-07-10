import * as planck from "planck";

const { World, Vec2, Circle, Box, Polygon } = planck;

// How two fixtures' restitutions combine into the contact's restitution. Box2D's
// built-in rule is max(); we override it: if both are positive, take the max,
// otherwise add them and clamp the result to >= 0. Applied per-contact via a
// pre-solve listener in create() (Planck's internal mix function isn't a public
// hook, so we recompute and setRestitution before each contact is solved).
function mixRestitution(r1, r2) {
    if (r1 > 0 && r2 > 0) return Math.max(r1, r2);
    return Math.max(0, r1 + r2);
}

// Thin wrapper around a Box2D (Planck.js) world. Units are meters; +y is down to
// match the Pixi screen, so gravity is positive-y. The fixed timestep is owned
// by Map (which calls step() once per tick); we advance the world by that dt,
// optionally split into a few substeps for stability.
//
// Bodies are wrapped in PhysicsBody so the rest of the game keeps the same
// Rapier-flavored interface (translation/rotation/linvel/mass/addForce/...) and
// doesn't care which engine is underneath. The raw Planck body is on `.body`.
export class PhysicsBody {
    constructor(body, { baseMass = 0 } = {}) {
        this.body = body;
        this.baseMass = baseMass; // mass from fixtures alone; setAdditionalMass stacks on this
    }

    translation() {
        const p = this.body.getPosition();
        return { x: p.x, y: p.y };
    }

    rotation() {
        return this.body.getAngle();
    }

    linvel() {
        const v = this.body.getLinearVelocity();
        return { x: v.x, y: v.y };
    }

    mass() {
        return this.body.getMass();
    }

    // Teleport the body (used by the RL env's reset). Wakes it so it re-simulates.
    setTranslation(p) {
        this.body.setPosition(new Vec2(p.x, p.y));
        this.body.setAwake(true);
    }

    setLinvel(v) {
        this.body.setLinearVelocity(new Vec2(v.x, v.y));
        this.body.setAngularVelocity(0);
    }

    // Box2D clears accumulated forces after every step (autoClearForces), so
    // there's nothing to reset between ticks — kept for interface parity.
    resetForces() {}

    addForce(f) {
        this.body.applyForceToCenter(new Vec2(f.x, f.y), true);
    }

    applyImpulse(i) {
        this.body.applyLinearImpulse(new Vec2(i.x, i.y), this.body.getWorldCenter(), true);
    }

    // Rapier had setAdditionalMass (stacked on the fixture mass). Emulate it by
    // overriding the body's mass to baseMass + extra, keeping the COM centered and
    // rotational inertia zero (the ball uses fixed rotation anyway).
    setAdditionalMass(extra) {
        this.body.setMassData({ mass: this.baseMass + extra, center: new Vec2(0, 0), I: 0 });
    }
}

export class Physics {
    constructor(world, { fixedDt, substeps, velocityIterations, positionIterations }) {
        this.planck = planck;
        this.world = world;
        this.fixedDt = fixedDt;
        this.substeps = substeps;
        this.velocityIterations = velocityIterations;
        this.positionIterations = positionIterations;
    }

    // Kept async so callers (Map.create) don't change; Planck needs no async init.
    static async create({
        gravity = { x: 0, y: 20 },
        fixedDt = 1 / 30,
        // lengthUnit is a Rapier concept (Box2D works in plain meters); accepted
        // and ignored so existing callers keep working.
        lengthUnit = 1,
        substeps = 1,
        velocityIterations = 2,
        positionIterations = 6,
    } = {}) {
        const world = new World(new Vec2(gravity.x, gravity.y));
        // Override Box2D's default max() restitution combine rule (see
        // mixRestitution). Pre-solve runs before each contact is solved, so this
        // sets the effective bounce for that step.
        world.on("pre-solve", (contact) => {
            const r1 = contact.getFixtureA().getRestitution();
            const r2 = contact.getFixtureB().getRestitution();
            contact.setRestitution(mixRestitution(r1, r2));
        });
        return new Physics(world, { fixedDt, substeps, velocityIterations, positionIterations });
    }

    // Convert one of a Shape's engine-neutral collision descriptors into a Planck
    // shape (in coordinates local to the body placed at the shape's origin).
    #toPlanckShape(cs) {
        switch (cs.type) {
            case "circle":
                return new Circle(new Vec2(cs.cx ?? 0, cs.cy ?? 0), cs.radius);
            case "box":
                return new Box(cs.hx, cs.hy, new Vec2(cs.cx ?? 0, cs.cy ?? 0), cs.angle ?? 0);
            case "polygon":
                return new Polygon(cs.vertices.map((v) => new Vec2(v.x, v.y)));
            default:
                throw new Error(`physics: unknown collision shape "${cs.type}"`);
        }
    }

    addPlatform(shape) {
        const body = this.world.createBody({ type: "static", position: new Vec2(shape.x, shape.y) });
        for (const cs of shape.collisionShapes()) {
            let planckShape;
            try {
                planckShape = this.#toPlanckShape(cs);
            } catch (e) {
                // Box2D caps polygons at 8 vertices and needs convex input; skip a
                // bad collider (the shape still renders) rather than crash the map.
                console.warn(`physics: skipping collider — ${e.message}`);
                continue;
            }
            body.createFixture({
                shape: planckShape,
                friction: shape.friction,
                restitution: shape.bounciness,
            });
        }
        return body;
    }

    // Dynamic circle (e.g. the player). Returns { body, collider } where body is
    // a PhysicsBody and collider is the Planck fixture (used to exclude self from
    // ground raycasts). Mass is pinned to `mass` regardless of radius.
    addBall({
        x,
        y,
        radius,
        mass = 1,
        friction = 0,
        // With the ball at 0, our mixRestitution rule falls to the "add" branch,
        // so a bouncy platform's restitution passes straight through and the ball
        // bounces off it while a plain 0-restitution platform stays dead.
        restitution = 0.95,
        linearDamping = 0,
        lockRotation = true,
    }) {
        const body = this.world.createBody({
            type: "dynamic",
            position: new Vec2(x, y),
            linearDamping,
            fixedRotation: lockRotation,
            bullet: true, // CCD, so a fast ball can't tunnel through platforms
        });
        const collider = body.createFixture({
            shape: new Circle(radius),
            density: 1, // real value comes from setMassData below
            friction,
            restitution,
        });
        // Pin the mass so it doesn't depend on the radius/density.
        body.setMassData({ mass, center: new Vec2(0, 0), I: 0 });
        return { body: new PhysicsBody(body, { baseMass: mass }), collider };
    }

    // Cast a ray and return the nearest hit ({ fixture, point, normal, fraction })
    // or null. `dir` should be a unit vector; `maxToi` is the max distance.
    // `excludeCollider` (a fixture) is skipped so a body can cast from inside
    // itself (e.g. a ground check).
    castRay(origin, dir, maxToi, { excludeCollider } = {}) {
        const p1 = new Vec2(origin.x, origin.y);
        const p2 = new Vec2(origin.x + dir.x * maxToi, origin.y + dir.y * maxToi);
        let hit = null;
        this.world.rayCast(p1, p2, (fixture, point, normal, fraction) => {
            if (excludeCollider && fixture === excludeCollider) return -1; // ignore, keep searching
            hit = {
                fixture,
                point: { x: point.x, y: point.y },
                normal: { x: normal.x, y: normal.y },
                fraction,
            };
            return fraction; // clip the search to the closest hit so far
        });
        return hit;
    }

    // Advance the world by one logical tick, split into `substeps` smaller steps.
    step() {
        const dt = this.fixedDt / this.substeps;
        for (let i = 0; i < this.substeps; i++) {
            this.world.step(dt, this.velocityIterations, this.positionIterations);
        }
    }
}
