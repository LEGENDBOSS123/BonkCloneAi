import { Container, Graphics } from "pixi.js";
import { createApp } from "./app.mjs";
import { Env } from "./env.mjs";
import { TICK_DT } from "./map/index.mjs";

// Demo: run the RL Env with random actions for BOTH players, render the world,
// and log each player's observation (its own POV) every step.
//
// The Env owns the simulation here (env.step advances the physics), so we do NOT
// use MapRenderingContext — that class steps the Map itself, which would
// double-step. Instead we build a minimal camera + views and, after each fixed
// tick, position the views straight from the body transforms.

const WORLD_HEIGHT = 64; // meters visible vertically

const app = await createApp();
const env = await Env.create({ gravity: 20 });
const map = env.map;
top.env = env; // handy for poking at it from the console

// --- Scene ------------------------------------------------------------------
// A scaled camera container holding all world views (meters -> screen pixels).
const camera = new Container();
app.stage.addChild(camera);
for (const shape of map.shapes) camera.addChild(shape.view);
for (const player of map.players) camera.addChild(player.view);

// Debug overlay: the kill line (y = 250/ppm) and kill circle (r = 850/ppm),
// matching the Env's death thresholds.
const overlay = new Graphics();
const stroke = { color: 0xff0000, pixelLine: true };
const ppm = map.ppm;
overlay.moveTo(-1e6, 250 / ppm).lineTo(1e6, 250 / ppm).stroke(stroke);
overlay.circle(0, 0, 850 / ppm).stroke(stroke);
camera.addChild(overlay);

// Fit WORLD_HEIGHT meters into the canvas height and center on player 1.
function centerOn(view) {
    const scale = app.screen.height / WORLD_HEIGHT;
    camera.scale.set(scale);
    if (!view) return;
    camera.x = app.screen.width / 2 - view.x * scale;
    camera.y = app.screen.height / 2 - view.y * scale;
}

// --- Random policy ----------------------------------------------------------
function randomAction() {
    return {
        up: Math.random() < 0.5,
        down: Math.random() < 0.2,
        left: Math.random() < 0.4,
        right: Math.random() < 0.4,
        heavy: Math.random() < 0.3,
    };
}

// Round for readable logs without hiding the values.
const fmt = (arr) => arr.map((n) => Number(n.toFixed(3)));

// --- Fixed-timestep loop ----------------------------------------------------
let accumulator = 0;
app.ticker.add((ticker) => {
    accumulator += Math.min(ticker.deltaMS / 1000, 0.25);

    let steps = 0;
    while (accumulator >= TICK_DT && steps < 5) {
        const actions = { p1: randomAction(), p2: randomAction() };
        const { state, reward, done } = env.step(actions);

        console.log(
            `tick ${map.tick} | p1 POV: [${fmt(state.p1)}] | p2 POV: [${fmt(state.p2)}] | reward:`,
            reward,
            `| done: ${done}`
        );

        if (done) {
            console.log(`--- episode over (reward ${JSON.stringify(reward)}), resetting ---`);
            env.reset();
        }

        accumulator -= TICK_DT;
        steps++;
    }

    // Render: place each player's view at its current body transform.
    for (const player of map.players) {
        const t = player.body.translation();
        player.view.x = t.x;
        player.view.y = t.y;
        player.view.rotation = player.body.rotation();
    }
    centerOn(map.player?.view ?? map.players[0]?.view);
});
