import { Container, Graphics } from "pixi.js";
import { createApp } from "../app.mjs";
import { parseBonkMap } from "../bonkmap/bonkmap_parser.mjs";

// Replay viewer for headless rl2 games. Loads map1.json for the scenery, then
// plays back a replay JSON (from headless_node.mjs) by driving the two players'
// views from the recorded frames. No physics runs here — pure playback, so what
// you see is exactly what the headless sim computed.
//
// Controls: pick file (button / drag-drop) · Space pause · ←/→ step ·
// R restart · +/- speed.

const app = await createApp();
const mapData = await (await fetch(new URL("../bonkmap/map1.json", import.meta.url))).json();
const map = await parseBonkMap(mapData, { gravity: 20 });

const camera = new Container();
app.stage.addChild(camera);
for (const shape of map.shapes) camera.addChild(shape.view);
for (const player of map.players) camera.addChild(player.view);

const overlay = new Graphics();
const stroke = { color: 0xff0000, pixelLine: true };
overlay.moveTo(-1e6, 250 / map.ppm).lineTo(1e6, 250 / map.ppm).stroke(stroke);
overlay.circle(0, 0, 850 / map.ppm).stroke(stroke);
camera.addChild(overlay);

const WORLD_HEIGHT = 52;
function layoutCamera() {
    const scale = app.screen.height / WORLD_HEIGHT;
    camera.scale.set(scale);
    camera.x = app.screen.width / 2;
    camera.y = app.screen.height / 2;
}
layoutCamera();

// --- state ------------------------------------------------------------------
let replay = null;   // { tps, result, frames }
let playhead = 0;    // fractional frame index
let playing = true;
let speed = 1;

function applyFrame(idx) {
    if (!replay) return;
    const f = replay.frames[Math.max(0, Math.min(replay.frames.length - 1, idx))];
    const p = map.players;
    p[0].view.position.set(f[0], f[1]);
    p[0].view.rotation = f[2];
    p[1].view.position.set(f[3], f[4]);
    p[1].view.rotation = f[5];
}

// --- HUD ------------------------------------------------------------------------
const hud = document.createElement("div");
Object.assign(hud.style, {
    position: "fixed", top: "8px", left: "8px", padding: "8px 10px",
    font: "12px/1.5 monospace", color: "#cfe", background: "rgba(0,0,0,0.55)",
    borderRadius: "6px", whiteSpace: "pre", pointerEvents: "none", zIndex: 10,
});
document.body.appendChild(hud);

function updateHud() {
    hud.textContent = replay
        ? [
            `frame  ${Math.floor(playhead)} / ${replay.frames.length - 1}`,
            `result ${replay.result}`,
            `speed  ${speed.toFixed(2)}x${playing ? "" : "  (PAUSED)"}`,
            `keys   Space pause · ←/→ step · R restart · +/- speed`,
        ].join("\n")
        : "Load a replay JSON (button top-right, or drop the file here).";
}

const loadBtn = document.createElement("button");
loadBtn.textContent = "Load replay";
Object.assign(loadBtn.style, {
    position: "fixed", top: "8px", right: "8px", zIndex: 10,
    font: "12px monospace", padding: "6px 10px", cursor: "pointer",
});
document.body.appendChild(loadBtn);

function loadReplay(json) {
    replay = json;
    playhead = 0;
    playing = true;
    applyFrame(0);
    console.log(`replay loaded: ${json.frames.length} frames, result ${json.result}`);
}

loadBtn.onclick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json,.json";
    input.onchange = async () => {
        const file = input.files?.[0];
        if (file) loadReplay(JSON.parse(await file.text()));
    };
    input.click();
};

window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", async (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) loadReplay(JSON.parse(await file.text()));
});

window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { playing = !playing; e.preventDefault(); }
    else if (e.code === "ArrowRight") { playing = false; playhead = Math.min(replay ? replay.frames.length - 1 : 0, Math.floor(playhead) + 1); }
    else if (e.code === "ArrowLeft") { playing = false; playhead = Math.max(0, Math.floor(playhead) - 1); }
    else if (e.code === "KeyR") { playhead = 0; playing = true; }
    else if (e.key === "+" || e.key === "=") speed = Math.min(16, speed * 2);
    else if (e.key === "-") speed = Math.max(0.125, speed / 2);
});

// --- playback loop ------------------------------------------------------------------
app.ticker.add((ticker) => {
    layoutCamera();
    if (replay && playing) {
        playhead += (ticker.deltaMS / 1000) * (replay.tps ?? 30) * speed;
        if (playhead >= replay.frames.length - 1) {
            playhead = replay.frames.length - 1;
            playing = false;
        }
    }
    applyFrame(Math.floor(playhead));
    updateHud();
});
updateHud();
