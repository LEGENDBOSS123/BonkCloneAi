import { Physics } from "../physics.mjs";

// The simulation runs at a fixed 30 ticks per second. Rendering happens at the
// display refresh rate and interpolates between ticks (see MapRenderingContext).
export const TPS = 30;
export const TICK_DT = 1 / TPS;

// A parkour map: the central object everything revolves around. It owns the
// physics world, the platform shapes, and the player, and advances the whole
// simulation one fixed tick at a time via step().
export class Map {
    constructor({ name = "Untitled", author = "Anonymous", gravity = 20, lengthUnit = 1 } = {}) {
        this.name = name;
        this.author = author;
        this.gravity = gravity;
        this.lengthUnit = lengthUnit; // world units are small (player radius ~1)
        this.shapes = [];
        this.players = [];
        this.player = null; // the primary controllable player, followed by the camera
        this.spawnPoints = []; // world-space spawn positions [{x, y}, ...] (set by the loader)
        this.physics = null;
        this.tick = 0;
        this.ppm = 1;
    }

    // Physics.create is async (kept so the engine can be swapped freely), so
    // maps are created asynchronously.
    static async create(opts = {}) {
        const map = new Map(opts);
        map.physics = await Physics.create({
            gravity: { x: 0, y: map.gravity },
            fixedDt: TICK_DT,
            lengthUnit: map.lengthUnit,
        });
        return map;
    }

    addShape(shape) {
        this.shapes.push(shape);
        // Decorative shapes (solid === false) are rendered but get no collider.
        if (shape.solid !== false) this.physics.addPlatform(shape);
        return shape;
    }

    // Add a player to the map, giving it a physics body. The first controllable
    // player becomes `this.player` (the one the camera follows).
    addPlayer(player) {
        const { body, collider } = this.physics.addBall({
            x: player.x,
            y: player.y,
            radius: player.radius,
        });
        player.attach({ body, collider, physics: this.physics });
        this.players.push(player);
        if (player.controllable && !this.player) this.player = player;
        return player;
    }

    // Back-compat single-player helper: adds one player and follows it.
    setPlayer(player) {
        return this.addPlayer(player);
    }

    // Advance the simulation by exactly one fixed tick. Pure simulation — the
    // renderer reads body transforms itself for interpolation.
    step(input) {
        for (const player of this.players) player.control(input);
        this.physics.step();
        this.tick++;
    }
}
