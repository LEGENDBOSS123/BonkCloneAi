import * as tf from "@tensorflow/tfjs";
import { Container, Graphics } from "pixi.js";
import { createApp } from "../app.mjs";
import { Env } from "../env.mjs";
import { CONFIG } from "./config.mjs";
import { LagEnv } from "./lag_env.mjs";
import { NFSPAgent } from "./agent.mjs";
import { Trainer } from "./trainer.mjs";

// rl2 browser entry: NFSP (discrete SAC best response + supervised average
// policy) with input-lag simulation. Same operational conveniences as rl1:
// WebGPU-first backend, HUD, top.gameSpeed, pause/save hotkeys, backup file
// load, and a Web Worker metronome so training survives background tabs.

const config = CONFIG;
const app = await createApp();
await initBackend();
console.log(`tfjs backend: ${tf.getBackend()}`);

async function initBackend() {
    const forced = new URLSearchParams(location.search).get("backend");
    if (forced) {
        if (forced === "webgpu") await import("@tensorflow/tfjs-backend-webgpu").catch(() => { });
        await tf.setBackend(forced).catch((e) => console.warn(`forced backend ${forced} failed:`, e?.message));
        await tf.ready();
        return;
    }
    if (navigator.gpu) {
        try {
            await import("@tensorflow/tfjs-backend-webgpu");
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
        } catch { /* try the next backend */ }
    }
    await tf.ready();
}

// --- Envs / agent / trainer ---------------------------------------------------
const numEnvs = config.training.numEnvs;
const mapData = await (await fetch(new URL("../bonkmap/map1.json", import.meta.url))).json();
const lagEnvs = [];
for (let i = 0; i < numEnvs; i++) {
    const core = await Env.create({ data: mapData, gravity: config.env.gravity });
    lagEnvs.push(new LagEnv(core, config));
}
const env0 = lagEnvs[0];
const agent = new NFSPAgent(env0.stateDim, config);
const trainer = new Trainer(lagEnvs, agent, config);
const map = env0.core.map;
console.log(`rl2/NFSP: ${numEnvs} envs, obs dim ${env0.stateDim}, ${config.numActions} actions, lag ${config.env.inputLag} ticks`);

top.rl2 = { agent, trainer, envs: lagEnvs, config };
top.saveProgress = saveProgress;
top.gameSpeed = config.render.defaultGameSpeed;

// --- Scene (render env 0) --------------------------------------------------------
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
    centerOn(map.players[0].view);
}

// --- HUD ----------------------------------------------------------------------
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

let paused = false;
let stepsPerSec = 0;
let rateStepRef = 0;
let rateTimeRef = performance.now();
let perfRef = { decideMs: 0, physMs: 0, trainMs: 0, ticks: 0 };
let perfLine = "perf  —";
const backend = tf.getBackend();
const backendWarn = backend === "cpu" ? "  ⚠ NO GPU — very slow!" : "";

function updateHud() {
    const now = performance.now();
    if (now - rateTimeRef > 500) {
        stepsPerSec = (trainer.envSteps - rateStepRef) / ((now - rateTimeRef) / 1000);
        rateStepRef = trainer.envSteps;
        rateTimeRef = now;
        // Per-tick phase breakdown since the last HUD refresh.
        const p = trainer.perf;
        const dt = Math.max(1, p.ticks - perfRef.ticks);
        perfLine = `perf/tick decide=${((p.decideMs - perfRef.decideMs) / dt).toFixed(2)}ms ` +
            `phys=${((p.physMs - perfRef.physMs) / dt).toFixed(2)}ms ` +
            `train=${((p.trainMs - perfRef.trainMs) / dt).toFixed(2)}ms`;
        perfRef = { ...p };
    }
    const s = agent.stats;
    const m = tf.memory();
    const mb = (b) => (b == null ? "?" : (b / 1048576).toFixed(0) + "MB");
    hud.textContent = [
        `backend   ${backend}${backendWarn}`,
        `episode   ${trainer.episodeCount} / ${config.training.totalEpisodes}`,
        `ELO(avg)  ${trainer.currentRating.toFixed(1)}  (eval wr ${(trainer.evalWinRate() * 100).toFixed(0)}%)`,
        `BR vs AVG ${(trainer.brWinRate() * 100).toFixed(1)}%  (~50% => hard to exploit)`,
        `replay    ${trainer.replay.size} / ${trainer.replay.capacity}`,
        `reservoir ${trainer.reservoir.size} / ${trainer.reservoir.capacity}`,
        `grad steps sac=${agent.sacSteps} sl=${agent.slSteps}`,
        `alpha     ${s.alpha.toFixed(4)}  H(br) ${s.entropy.toFixed(3)} / ${agent.targetEntropy.toFixed(3)}`,
        `loss  q=${s.qLoss.toFixed(4)} pi=${s.policyLoss.toFixed(4)} avgCE=${s.avgLoss.toFixed(4)}`,
        `snapshots ${trainer.buffer.size}`,
        `steps/s   ${stepsPerSec.toFixed(0)}`,
        perfLine,
        `gameSpeed ${Number(top.gameSpeed).toFixed(2)}${paused ? "  (PAUSED)" : ""}`,
        `mem  tensors=${m.numTensors} bytes=${mb(m.numBytes)} gpu=${mb(m.numBytesInGPU)}`,
    ].join("\n");
}

// --- Keyboard: Space = pause, S = save -------------------------------------------
window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { paused = !paused; e.preventDefault(); }
    else if (e.code === "KeyS") saveProgress();
});

// --- Save / load --------------------------------------------------------------------
async function saveProgress() {
    // Halt training chunks while serializing so the checkpoint is a consistent
    // snapshot (all nets from the same instant, no mid-save grad steps).
    const wasPaused = paused;
    paused = true;
    let json;
    try {
        json = JSON.stringify(await trainer.serialize());
    } finally {
        paused = wasPaused;
    }
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bonk-nfsp-ep${trainer.episodeCount}-elo${Math.round(trainer.currentRating)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`Saved rl2 progress at episode ${trainer.episodeCount}.`);
}

function pickFile() {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "application/json,.json";
        let settled = false;
        const done = (v) => { if (!settled) { settled = true; resolve(v); } };
        input.onchange = () => done(input.files?.[0] ?? null);
        window.addEventListener("focus", () => setTimeout(() => done(null), 500), { once: true });
        input.click();
    });
}

async function maybeLoadBackup() {
    const wants = window.confirm(
        "Continue from a saved rl2 backup?\n\nOK = choose a backup .json file\nCancel = start fresh"
    );
    if (!wants) return;
    const file = await pickFile();
    if (!file) { console.log("No backup selected; starting fresh."); return; }
    try {
        trainer.loadState(JSON.parse(await file.text()));
        console.log(`Loaded rl2 backup: episode ${trainer.episodeCount}, ELO ${trainer.currentRating.toFixed(1)}.`);
    } catch (err) {
        console.error("Failed to load backup:", err);
        window.alert(`Failed to load backup: ${err.message}`);
    }
}

// --- Startup + main loop --------------------------------------------------------------
await maybeLoadBackup();
render();
updateHud();

const ticksPerSecondBase = config.render.ticksPerSecondBase;
const maxTicksPerChunk = config.render.maxTicksPerChunk;
let tickAccumulator = 0;
let lastChunkTime = performance.now();

app.ticker.add(() => {
    render();
    updateHud();
});

async function trainingChunk() {
    const now = performance.now();
    const dt = Math.min(0.25, (now - lastChunkTime) / 1000);
    lastChunkTime = now;

    const finished = trainer.episodeCount >= config.training.totalEpisodes;
    if (paused || finished) return;

    tickAccumulator += dt * (Number(top.gameSpeed) || 0) * ticksPerSecondBase;
    // Clamp the backlog so lowering gameSpeed takes effect immediately.
    tickAccumulator = Math.min(tickAccumulator, maxTicksPerChunk * 4);
    let ticks = Math.min(maxTicksPerChunk, Math.floor(tickAccumulator));
    tickAccumulator -= ticks;

    while (ticks-- > 0) {
        const infos = await trainer.tick();   // collect (decisions + physics)
        trainer.trainTick();                  // interleaved gradient steps
        for (const info of infos) {
            if (info.episode % config.training.logIntervalEpisodes === 0) {
                const s = agent.stats;
                console.log(
                    `ep ${info.episode} | ELO ${trainer.currentRating.toFixed(1)} | ` +
                    `BRvAVG ${(trainer.brWinRate() * 100).toFixed(1)}% | ` +
                    `q=${s.qLoss.toFixed(4)} avgCE=${s.avgLoss.toFixed(4)} H=${s.entropy.toFixed(3)} a=${s.alpha.toFixed(4)}`
                );
            }
        }
    }
}

// Web Worker metronome: keeps ticking when the tab is hidden (page timers are
// throttled there, worker timers are not). Ping-pong prevents overlap.
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
        const loop = async () => { await trainingChunk().catch((e) => console.error(e)); setTimeout(loop, 0); };
        loop();
    }
}
startMetronome();
