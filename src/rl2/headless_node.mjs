import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as tf from "@tensorflow/tfjs";
import { Env } from "../env.mjs";
import { CONFIG } from "./config.mjs";
import { LagEnv } from "./lag_env.mjs";
import { NFSPAgent } from "./agent.mjs";
import { Trainer } from "./trainer.mjs";

// Headless rl2 (NFSP) training in Node — no browser, no rendering, no tab
// throttling. Same Trainer/Agent/config as train2.html; checkpoints are the
// SAME JSON format, so files move freely between browser and Node runs.
//
//   node src/rl2/headless_node.mjs [options]
//
// Options:
//   --backend=wasm|cpu|node   tfjs backend (default wasm; "node" tries
//                             @tensorflow/tfjs-node if installed)
//   --load=FILE               resume from a checkpoint JSON
//   --episodes=N              stop after N total episodes (default config)
//   --save-every=N            checkpoint every N episodes (default 2000)
//   --replay-every=N          record env-0 episode replay every N episodes
//                             (default 500; 1 = every env-0 episode)
//   --out=DIR                 output dir (default ./runs)
//   --learn-start=N           override config.nfsp.learnStart (quick tests)
//
// Replays land in DIR/replays/ep<N>.json — open replay.html to watch one.

const args = Object.fromEntries(
    process.argv.slice(2).map((a) => {
        const m = a.match(/^--([^=]+)(?:=(.*))?$/);
        return m ? [m[1], m[2] ?? true] : [a, true];
    })
);

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(args.out ?? "runs");
const replayDir = path.join(outDir, "replays");
fs.mkdirSync(replayDir, { recursive: true });

// --- backend ------------------------------------------------------------------
async function initBackend() {
    const want = args.backend ?? "wasm";
    if (want === "node") {
        try {
            // tfjs-node 4.x calls util.isNullOrUndefined / isNull etc., which
            // Node >= 23 removed from the builtin. Restore them before loading.
            const util = (await import("node:util")).default;
            util.isNullOrUndefined ??= (v) => v === null || v === undefined;
            util.isNull ??= (v) => v === null;
            util.isUndefined ??= (v) => v === undefined;
            await import("@tensorflow/tfjs-node"); // registers "tensorflow"
            await tf.setBackend("tensorflow");
            await tf.ready();
            return;
        } catch (err) {
            console.warn("tfjs-node unavailable (npm i @tensorflow/tfjs-node); falling back to wasm:", err?.message);
        }
    }
    if (want === "wasm" || want === "node") {
        try {
            await import("@tensorflow/tfjs-backend-wasm");
            await tf.setBackend("wasm");
            await tf.ready();
            return;
        } catch (err) {
            console.warn("wasm backend unavailable; falling back to cpu:", err?.message);
        }
    }
    await tf.setBackend("cpu");
    await tf.ready();
}
await initBackend();
console.log(`tfjs backend: ${tf.getBackend()}`);

// --- config overrides ------------------------------------------------------------
const config = CONFIG;
if (args["learn-start"]) config.nfsp.learnStart = Number(args["learn-start"]);
const totalEpisodes = args.episodes ? Number(args.episodes) : config.training.totalEpisodes;
const saveEvery = Number(args["save-every"] ?? 2000);
const replayEvery = Number(args["replay-every"] ?? 500);

// --- envs / agent / trainer -------------------------------------------------------
const mapData = JSON.parse(fs.readFileSync(path.join(here, "../bonkmap/map1.json"), "utf8"));
const lagEnvs = [];
for (let i = 0; i < config.training.numEnvs; i++) {
    const core = await Env.create({ data: mapData, gravity: config.env.gravity });
    lagEnvs.push(new LagEnv(core, config));
}
const agent = new NFSPAgent(lagEnvs[0].stateDim, config);
const trainer = new Trainer(lagEnvs, agent, config);
console.log(`rl2/NFSP headless: ${config.training.numEnvs} envs, obs ${lagEnvs[0].stateDim}, lag ${config.env.inputLag}, net [${config.network.hidden}]`);

if (args.load) {
    trainer.loadState(JSON.parse(fs.readFileSync(String(args.load), "utf8")));
    console.log(`resumed: episode ${trainer.episodeCount}, ELO ${trainer.currentRating.toFixed(1)}`);
}

// --- replay recorder (observes env 0 via the trainer hook) --------------------------
// File: { version, tps, actionRepeat, inputLag, result, frames } where each
// frame is [x0, y0, rot0, x1, y1, rot1, a0, a1] (positions in world meters,
// a* = applied joint-action indices). Positions are post-tick.
const r3 = (v) => Math.round(v * 1000) / 1000;
trainer.recorder = {
    frames: [],
    episodesSeen: 0,
    frame(env, res) {
        const p = env.core.map.players;
        const [a0, a1] = env.appliedActions();
        const t0 = p[0].body.translation();
        const t1 = p[1].body.translation();
        this.frames.push([
            r3(t0.x), r3(t0.y), r3(p[0].body.rotation()),
            r3(t1.x), r3(t1.y), r3(p[1].body.rotation()),
            a0, a1,
        ]);
        if (!res.done) return;
        this.episodesSeen++;
        if (this.episodesSeen % replayEvery === 0) {
            const result = res.timeout || (res.dead[0] && res.dead[1])
                ? "draw"
                : res.dead[1] ? "p1-wins" : "p2-wins";
            const file = path.join(replayDir, `ep${trainer.episodeCount + 1}-${result}.json`);
            fs.writeFileSync(file, JSON.stringify({
                version: 1,
                tps: 30,
                actionRepeat: config.training.actionRepeat,
                inputLag: config.env.inputLag,
                result,
                frames: this.frames,
            }));
            console.log(`replay saved: ${file} (${this.frames.length} frames, ${result})`);
        }
        this.frames = [];
    },
};

// --- checkpointing -----------------------------------------------------------------
let saving = false;
async function saveCheckpoint(tag) {
    if (saving) return;
    saving = true;
    try {
        const file = path.join(outDir, `bonk-nfsp-ep${trainer.episodeCount}-elo${Math.round(trainer.currentRating)}${tag ? `-${tag}` : ""}.json`);
        fs.writeFileSync(file, JSON.stringify(await trainer.serialize()));
        console.log(`checkpoint saved: ${file}`);
    } finally {
        saving = false;
    }
}

let stopRequested = false;
process.on("SIGINT", () => {
    if (stopRequested) process.exit(1); // second ^C: force quit
    stopRequested = true;
    console.log("\nSIGINT — saving checkpoint, then exiting (^C again to force).");
});

// --- main loop -----------------------------------------------------------------------
const t0 = performance.now();
let lastLog = t0;
let lastSteps = 0;
let lastPerf = { ...trainer.perf };
let lastSavedEpisode = trainer.episodeCount;

while (trainer.episodeCount < totalEpisodes && !stopRequested) {
    await trainer.tick();
    trainer.trainTick();

    const now = performance.now();
    if (now - lastLog >= 5000) {
        const p = trainer.perf;
        const dt = Math.max(1, p.ticks - lastPerf.ticks);
        const sps = ((trainer.envSteps - lastSteps) / ((now - lastLog) / 1000)).toFixed(0);
        console.log(
            `ep ${trainer.episodeCount} | steps/s ${sps} | ELO ${trainer.currentRating.toFixed(1)} | ` +
            `BRvAVG ${(trainer.brWinRate() * 100).toFixed(1)}% | replay ${trainer.replay.size} | ` +
            `sac ${agent.sacSteps} | H ${agent.stats.entropy.toFixed(3)} a ${agent.stats.alpha.toFixed(4)} | ` +
            `perf/tick decide=${((p.decideMs - lastPerf.decideMs) / dt).toFixed(2)} ` +
            `phys=${((p.physMs - lastPerf.physMs) / dt).toFixed(2)} ` +
            `train=${((p.trainMs - lastPerf.trainMs) / dt).toFixed(2)}ms`
        );
        lastLog = now;
        lastSteps = trainer.envSteps;
        lastPerf = { ...p };
    }

    if (trainer.episodeCount - lastSavedEpisode >= saveEvery) {
        lastSavedEpisode = trainer.episodeCount;
        await saveCheckpoint();
    }
}

await saveCheckpoint("final");
console.log(`done: ${trainer.episodeCount} episodes in ${((performance.now() - t0) / 60000).toFixed(1)} min.`);
process.exit(0);
