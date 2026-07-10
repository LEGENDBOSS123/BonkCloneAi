import * as tf from "@tensorflow/tfjs";
import { Container, Graphics } from "pixi.js";
import { createApp } from "../app.mjs";
import { Env } from "../env.mjs";
import { PPOAgent } from "./agent.mjs";
import { Trainer } from "./trainer.mjs";
import { CONFIG } from "../config.mjs";

const ACTION_DIM = 5; // up, down, left, right, heavy

const config = CONFIG;
const app = await createApp();
await initBackend();
console.log(`tfjs backend: ${tf.getBackend()}`);

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
await sleep(1000); // let the browser settle before we start logging

// Prefer WebGPU when the browser and hardware support it; otherwise fall back to
// WebGL, then CPU. The WebGPU backend is loaded lazily so a browser without it
// (or without navigator.gpu) still runs. Force a specific backend with a URL
// param, e.g. train.html?backend=webgl (handy if a WebGPU kernel is missing).
async function initBackend() {
    let forced = new URLSearchParams(location.search).get("backend");
    // forced = "webgl";
    if (forced) {
        if (forced === "webgpu") await import("@tensorflow/tfjs-backend-webgpu").catch(() => { });
        await tf.setBackend(forced).catch((e) => console.warn(`forced backend ${forced} failed:`, e?.message));
        await tf.ready();
        return;
    }
    if (navigator.gpu) {
        try {
            await import("@tensorflow/tfjs-backend-webgpu"); // registers the "webgpu" backend
            await tf.setBackend("webgpu");
            await tf.ready();
            if (tf.getBackend() === "webgpu") return;
        } catch (err) {
            console.warn("WebGPU backend unavailable, falling back:", err?.message ?? err);
        }
    }
    for (const backend of ["webgl", "cpu"]) {
        try {
            await tf.setBackend(backend);
            await tf.ready();
            if (tf.getBackend() === backend) return;
        } catch {
            /* try the next backend */
        }
    }
    await tf.ready();
}

// --- Env / agent / trainer --------------------------------------------------
// N parallel envs from a single map fetch. Each tick steps all of them with one
// batched forward/readback (see Trainer.collectBatch).
const numEnvs = config.training.numEnvs ?? 16;
const mapData = await (await fetch(new URL("../bonkmap/map1.json", import.meta.url))).json();
const envs = [];
for (let i = 0; i < numEnvs; i++) {
    envs.push(await Env.create({ data: mapData, gravity: config.env.gravity, shaping: config.env.shaping }));
}
const env = envs[0]; // the rendered env
const agent = new PPOAgent(env.stateSize, ACTION_DIM, config);
const trainer = new Trainer(envs, agent, config);
const map = env.map;
console.log(`training with ${numEnvs} parallel envs, obs dim ${env.stateSize}`);

top.envs = envs;
top.env = env;
top.agent = agent;
top.trainer = trainer;
top.saveProgress = saveProgress;
top.gameSpeed = config.render.defaultGameSpeed; // change live to speed up / slow down

// --- Scene (same rendering approach as the demo; Env owns stepping) ----------
const camera = new Container();
app.stage.addChild(camera);
for (const shape of map.shapes) camera.addChild(shape.view);
for (const player of map.players) camera.addChild(player.view);

const overlay = new Graphics();
const stroke = { color: 0xff0000, pixelLine: true };
const ppm = map.ppm;
overlay.moveTo(-1e6, 250 / ppm).lineTo(1e6, 250 / ppm).stroke(stroke);
overlay.circle(0, 0, 850 / ppm).stroke(stroke);
camera.addChild(overlay);

function centerOn(view) {
    const scale = app.screen.height / config.render.worldHeight;
    camera.scale.set(scale);
    if (!view) return;
    camera.x = app.screen.width / 2 - view.x * scale;
    camera.y = app.screen.height / 2 - view.y * scale;
}

function render() {
    for (const player of map.players) {
        const t = player.body.translation();
        player.view.x = t.x;
        player.view.y = t.y;
        player.view.rotation = player.body.rotation();
    }
    centerOn(map.players[trainer.learnerIndex].view);
}

// --- HUD --------------------------------------------------------------------
const hud = document.createElement("div");
Object.assign(hud.style, {
    position: "fixed", top: "8px", left: "8px", padding: "8px 10px",
    font: "12px/1.5 monospace", color: "#cfe", background: "rgba(0,0,0,0.55)",
    borderRadius: "6px", whiteSpace: "pre", pointerEvents: "none", zIndex: 10,
});
document.body.appendChild(hud);

const saveBtn = document.createElement("button");
saveBtn.textContent = "Save progress";
Object.assign(saveBtn.style, {
    position: "fixed", top: "8px", right: "8px", zIndex: 10,
    font: "12px monospace", padding: "6px 10px", cursor: "pointer",
});
saveBtn.onclick = saveProgress;
document.body.appendChild(saveBtn);

let lastStats = null;
let paused = false;
let isUpdating = false;    // true while a PPO update is running
let lastUpdateMs = 0;      // wall time of the most recent PPO update
let envStepsTotal = 0;     // cumulative env steps (numEnvs per tick)
let stepsPerSec = 0;
let rateStepRef = 0;
let rateTimeRef = performance.now();

const backend = tf.getBackend();
const backendWarn = backend === "cpu" ? "  ⚠ NO GPU — very slow!" : "";

function updateHud() {
    const now = performance.now();
    if (now - rateTimeRef > 500) {
        stepsPerSec = (envStepsTotal - rateStepRef) / ((now - rateTimeRef) / 1000);
        rateStepRef = envStepsTotal;
        rateTimeRef = now;
    }
    const winPct = (trainer.winRate() * 100).toFixed(1);
    const mix = trainer.opponentMix();
    hud.textContent = [
        `backend   ${backend}${backendWarn}`,
        `episode   ${trainer.episodeCount} / ${config.training.totalEpisodes}`,
        `updates   ${trainer.updateCount}`,
        `envs      ${trainer.numEnvs}  (${mix.current} cur / ${mix.snapshot} snap)`,
        `ELO       ${trainer.currentRating.toFixed(1)}`,
        `win-rate  ${winPct}%  (last ${trainer.recentResults.length})`,
        `snapshots ${trainer.buffer.size}`,
        `rollout   ${trainer.learnerSteps()} / ${config.ppo.rolloutSteps} (+${trainer.opponentSteps()} opp)`,
        `steps/s   ${stepsPerSec.toFixed(0)}`,
        `updateMs  ${lastUpdateMs.toFixed(0)}${isUpdating ? "  (updating…)" : ""}`,
        `gameSpeed ${Number(top.gameSpeed).toFixed(2)}${paused ? "  (PAUSED)" : ""}`,
        lastStats
            ? `loss  a=${lastStats.actorLoss.toFixed(3)} c=${lastStats.criticLoss.toFixed(3)} H=${lastStats.entropy.toFixed(3)}`
            : "loss  —",
        (() => {
            const m = tf.memory();
            const mb = (b) => (b == null ? "?" : (b / 1048576).toFixed(0) + "MB");
            // numBytes = tf-tracked CPU-side; numBytesInGPU (webgpu) = GPU buffers
            // tf knows about. If neither grows while the GPU process does, the leak
            // is in the backend's untracked buffers (not our tensors).
            return `mem  tensors=${m.numTensors} buffers=${m.numDataBuffers ?? "?"} ` +
                `bytes=${mb(m.numBytes)} gpu=${mb(m.numBytesInGPU)}`;
        })(),
    ].join("\n");
}

// --- Keyboard: Space = pause, S = save --------------------------------------
window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { paused = !paused; e.preventDefault(); }
    else if (e.code === "KeyS") saveProgress();
});

// --- Save / load ------------------------------------------------------------
async function saveProgress() {
    const blob = new Blob([JSON.stringify(await trainer.serialize())], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bonk-ppo-ep${trainer.episodeCount}-elo${Math.round(trainer.currentRating)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`Saved progress at episode ${trainer.episodeCount}.`);
}

// Prompt for a backup file. Resolves to a File or null (also on cancel).
function pickFile() {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "application/json,.json";
        let settled = false;
        const done = (v) => { if (!settled) { settled = true; resolve(v); } };
        input.onchange = () => done(input.files?.[0] ?? null);
        // If the OS dialog is canceled, onchange never fires; when focus returns
        // to the window, resolve null (after a beat so a real pick wins the race).
        window.addEventListener("focus", () => setTimeout(() => done(null), 500), { once: true });
        input.click();
    });
}

async function maybeLoadBackup() {
    const wants = window.confirm(
        "Continue from a saved backup?\n\nOK = choose a backup .json file\nCancel = start fresh"
    );
    if (!wants) return;
    const file = await pickFile();
    if (!file) { console.log("No backup selected; starting fresh."); return; }
    try {
        trainer.loadState(JSON.parse(await file.text()));
        console.log(`Loaded backup: episode ${trainer.episodeCount}, ELO ${trainer.currentRating.toFixed(1)}, snapshots ${trainer.buffer.size}.`);
    } catch (err) {
        console.error("Failed to load backup:", err);
        window.alert(`Failed to load backup: ${err.message}`);
    }
}

// --- Startup + main loop ----------------------------------------------------
await maybeLoadBackup();
render();
updateHud();

const ticksPerSecondBase = config.render.ticksPerSecondBase ?? 30;
const maxTicksPerChunk = config.render.maxTicksPerChunk ?? 8;
let tickAccumulator = 0;
let lastChunkTime = performance.now();

// Rendering runs on the animation frame (fine to pause when the tab is hidden).
// Training is driven separately by a metronome so it keeps going in a background
// tab and never blocks the renderer.
app.ticker.add(() => {
    render();
    updateHud();
});

// One training "chunk": run a real-time-paced, capped number of vectorized ticks
// (each advances ALL envs), then a PPO update if a rollout filled up. Kept short
// so it yields frequently — no long UI freezes even at high gameSpeed.
async function trainingChunk() {
    const now = performance.now();
    // Clamp dt so a long pause/update/hitch doesn't dump a huge budget at once.
    const dt = Math.min(0.25, (now - lastChunkTime) / 1000);
    lastChunkTime = now;

    const finished = trainer.episodeCount >= config.training.totalEpisodes;
    if (paused || isUpdating || finished) return;

    // Ticks this chunk = elapsed seconds * gameSpeed * base rate, accumulated so
    // fractional (slow-mo) speeds work; capped so each chunk stays responsive.
    tickAccumulator += dt * (Number(top.gameSpeed) || 0) * ticksPerSecondBase;
    let ticks = Math.min(maxTicksPerChunk, Math.floor(tickAccumulator));
    tickAccumulator -= ticks;

    while (ticks-- > 0) {
        const infos = await trainer.collectBatch();
        envStepsTotal += trainer.numEnvs;
        for (const info of infos) onEpisodeEnd(info);
        if (trainer.needsUpdate()) break;
    }

    if (trainer.needsUpdate()) {
        isUpdating = true;
        const t0 = performance.now();
        try {
            lastStats = await trainer.runUpdate();
        } catch (err) {
            console.error("PPO update failed:", err);
        } finally {
            lastUpdateMs = performance.now() - t0;
            isUpdating = false;
            lastChunkTime = performance.now(); // don't count the update time as elapsed
            console.log(`update ${trainer.updateCount}: ${lastUpdateMs.toFixed(0)}ms on ${backend}`);
        }
    }
}

// Drive trainingChunk from a Web Worker metronome. Worker timers aren't throttled
// when the tab is in the background (unlike requestAnimationFrame and page
// setTimeout, which are), so training keeps running while the tab is hidden. The
// worker only keeps time — all tf/env work stays on the main thread. Ping-pong
// (we post "next" only after each chunk resolves) prevents overlapping chunks.
function startMetronome() {
    const src = `onmessage=(e)=>{if(e.data==='start')postMessage(0);else if(e.data==='next')setTimeout(()=>postMessage(0),0);}`;
    if (typeof Worker !== "undefined") {
        const worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
        worker.onmessage = async () => {
            try { await trainingChunk(); }
            catch (err) { console.error("training chunk failed:", err); }
            worker.postMessage("next");
        };
        worker.postMessage("start");
    } else {
        // Fallback: page setTimeout (throttled to ~1s when hidden, but still runs).
        const loop = async () => { await trainingChunk().catch((e) => console.error(e)); setTimeout(loop, 0); };
        loop();
    }
}
startMetronome();

function onEpisodeEnd(info) {
    if (info.episode % config.training.logIntervalEpisodes === 0) {
        console.log(
            `ep ${info.episode} | ELO ${trainer.currentRating.toFixed(1)} | ` +
            `win-rate ${(trainer.winRate() * 100).toFixed(1)}% | snapshots ${trainer.buffer.size}` +
            (lastStats ? ` | loss a=${lastStats.actorLoss.toFixed(3)} c=${lastStats.criticLoss.toFixed(3)} H=${lastStats.entropy.toFixed(3)}` : "")
        );
    }
}
