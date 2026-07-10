import { Container, Graphics } from "pixi.js";
import { TICK_DT } from "./map.mjs";

// Bridges a Map's fixed-timestep simulation to a variable-rate render loop, and
// owns the camera.
//
// The world (shapes + player) lives in meters. The camera shows a fixed slice
// of the world that is `worldHeight` meters tall; the horizontal extent follows
// from the canvas aspect ratio. There is no fixed pixels-per-unit: the scale is
// derived each frame from screenHeight / worldHeight, so it stays correct for
// whatever pixel size the canvas happens to be.
//
// A "camera" Container holds all world views (positioned in meters); its scale
// and position turn those meters into screen pixels, recentered on the player.
//
// It also runs the fixed 30 TPS step loop and (optionally) interpolates dynamic
// entities between ticks. Interpolation state lives here; entities only expose a
// physics `body` and a Pixi `view`.
export class MapRenderingContext {
    constructor(map, { stage, screen, worldHeight = 8, interpolate = false, follow = true }) {
        this.map = map;
        this.screen = screen; // app.screen: { width, height } in pixels
        this.worldHeight = worldHeight; // meters of world visible vertically
        this.interpolate = interpolate;
        this.follow = follow;
        this.accumulator = 0;

        // The camera: a scaled container holding all world views.
        this.camera = new Container();
        stage.addChild(this.camera);

        // Mount every shape's view, then the players' on top.
        for (const shape of map.shapes) this.camera.addChild(shape.view);
        for (const player of map.players) this.camera.addChild(player.view);

        // Debug overlay (world space): a horizontal reference line and a circle
        // outline. `pixelLine` keeps the stroke 1px thin regardless of zoom.
        this.camera.addChild(this.#buildDebugOverlay());

        // One interpolation record per dynamic (body-driven) entity.
        this.dynamics = this.#collectDynamics();
        this.#centerOn(map.player?.view);
    }

    // A thin red horizontal line at y=500, a red circle outline (radius 850), and
    // a red box outline (730 x 500), all centered on the origin and drawn in
    // world coordinates (bonk pixels scaled by 1/ppm).
    #buildDebugOverlay() {
        const g = new Graphics();
        const stroke = { color: 0xff0000, pixelLine: true };
        const ppm = this.map.ppm;
        g.moveTo(-1e6, 250/ppm).lineTo(1e6, 250/ppm).stroke(stroke);
        g.circle(0, 0, 850/ppm).stroke(stroke);
        const hw = 730 / 2 / ppm;
        const hh = 500 / 2 / ppm;
        g.rect(-hw, -hh, hw * 2, hh * 2).stroke(stroke);
        return g;
    }

    // Pixels per meter: fit worldHeight meters into the canvas height.
    #scale() {
        return this.screen.height / this.worldHeight;
    }

    #collectDynamics() {
        const entities = [...this.map.players];
        return entities.map((entity) => {
            const snap = this.#readTransform(entity);
            return { entity, prev: { ...snap }, cur: { ...snap } };
        });
    }

    #readTransform(entity) {
        const t = entity.body.translation();
        return { x: t.x, y: t.y, rot: entity.body.rotation() };
    }

    // Scale and position the camera so the given (meter-space) view sits at the
    // viewport center: screen = camera.pos + viewPos * scale.
    #centerOn(view) {
        const scale = this.#scale();
        this.camera.scale.set(scale);
        if (!view || !this.follow) return;
        this.camera.x = this.screen.width / 2 - view.x * scale;
        this.camera.y = this.screen.height / 2 - view.y * scale;
    }

    // Call once per render frame with real elapsed seconds and the input source.
    // Runs as many fixed ticks as have accumulated (capped to avoid a death
    // spiral after a stall), then renders the leftover fraction as alpha.
    update(dt, input) {
        this.accumulator += Math.min(dt, 0.25);

        let ticks = 0;
        while (this.accumulator >= TICK_DT && ticks < 5) {
            for (const d of this.dynamics) {
                d.prev.x = d.cur.x;
                d.prev.y = d.cur.y;
                d.prev.rot = d.cur.rot;
            }
            this.map.step(input);
            for (const d of this.dynamics) {
                const t = this.#readTransform(d.entity);
                d.cur.x = t.x;
                d.cur.y = t.y;
                d.cur.rot = t.rot;
            }
            this.accumulator -= TICK_DT;
            ticks++;
        }

        // alpha is clamped to [0, 1] so a frame that hits the tick cap can't
        // extrapolate past the latest physics transform (which looks like phasing).
        const alpha = this.interpolate ? Math.min(this.accumulator / TICK_DT, 1) : 1;
        for (const { entity, prev, cur } of this.dynamics) {
            entity.view.x = prev.x + (cur.x - prev.x) * alpha;
            entity.view.y = prev.y + (cur.y - prev.y) * alpha;
            entity.view.rotation = prev.rot + (cur.rot - prev.rot) * alpha;
        }

        // Follow the player after its view position is finalized for this frame.
        this.#centerOn(this.map.player?.view);
    }
}
