// play3.mjs — paste-into-the-tab AI player for bonk2 (v2) checkpoints
// (python -m bonk2.train). Prompts for the checkpoint JSON, rebuilds the actor
// with TensorFlow.js, reads live bonk.io state, and drives your player.
//
// REQUIREMENTS (same as play2.mjs)
//   1. The bonk.io instrumentation from Bonk1v1Ai must already be injected:
//      top.playerids, top.myid, top.scale, top.presskeys, top.MAKE_KEYS,
//      top.GET_KEYS, top.RECIEVEFUNCTION (and ideally top.getCurrentFrame).
//   2. Network access (loads @tensorflow/tfjs from jsDelivr once).
//   3. Paste AFTER the instrumentation; pick the checkpoint in the file dialog.
//      Stop with top.bonkai.playStop().
//
// OBSERVATION must match python/bonk2/env.py, 34 dims:
//   [ 0-11] self : x,y,vx,vy, heavyValue(masked), up,down,left,right,heavy,
//                  lastHeavySeen, ticksSinceSeen/200 (cap 1)
//   [12-23] opp  : same, with the opponent's APPLIED keys
//   [24-27] rel  : dx,dy,dvx,dvy (normalized)
//   [28-32] pend : the keys WE LAST SENT (may not have applied yet — real lag!)
//   [33]    time : ticks since round start / MAX_EPISODE_STEPS
// Policy outputs 18 logits over joint actions: index = lr*6 + ud*2 + heavy.
//
// v2 heavy tracker: the heavy meter is only readable (sprite alpha) while the
// key is held; it regenerates unseen at 5/tick otherwise. We remember, per
// player, the last value seen and when — exactly what training saw.

(function () {
    "use strict";
    top.bonkai = top.bonkai || {};

    // ===== Coordinate conversion (calibrated in play.mjs — see its notes) ====
    const PPM = 15;                    // map1.json physics.ppm
    const ORIGIN_X = 365, ORIGIN_Y = 250; // live map-coords of the map center
    // bonk's playerData2.xvel is a finite difference in position-units per
    // MILLISECOND, so m/s = xvel * 1000 / (top.scale * PPM).
    const VEL_TIME = 1000;
    top.VEL_TIME = VEL_TIME;

    // ===== bonk2 env constants (python/bonk2/config.py — keep in sync!) ======
    const POS_SCALE = 1 / 30;
    const VEL_SCALE = 1 / 30;
    const MAX_EPISODE_STEPS = 3000;    // draw-clock horizon (ticks)
    const ACTION_REPEAT = 2;           // decide every 2 frames, hold between
    const HEAVY_SEEN_TICKS_NORM = 200; // full-regen span (5/tick from 0)
    const STATE_DIM = 34;
    const TPS = 30;
    const DECIDE_MS = (1000 / TPS) * ACTION_REPEAT;
    const GREEDY = true;               // argmax (set false to sample the policy)

    // Heavy charge: bonk sprite alpha in [0,1] == heavyPower/1000.
    const HEAVY_CHILD = 2;             // sprite child holding the alpha
    const HEAVY_INVERT = false;        // true if FULL charge reads as alpha 0

    const posX = (px) => (px / top.scale - ORIGIN_X) / PPM;
    const posY = (py) => (py / top.scale - ORIGIN_Y) / PPM;
    const velScale = () => top.VEL_TIME / (top.scale * PPM);

    // ===== Joint action index <-> keys (python/bonk/sim.py index_to_keys) ====
    function indexToKeys(index) {
        const lr = (index / 6) | 0;
        const ud = ((index / 2) | 0) % 3;
        return {
            left: lr === 1, right: lr === 2,
            up: ud === 1, down: ud === 2,
            heavy: index % 2 === 1, special: false,
        };
    }

    // ===== Local state =========================================================
    const keyMap = new Map();          // playerId -> encoded key int (from packets)
    const heavyTrack = new Map();      // playerId -> { seen: 0..1, seenAtTick }
    let tf = null;
    let policy = null;
    let running = false;
    let lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
    let lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 }; // pending block
    let roundStartFrame = null;        // for the draw clock + heavy tracker
    let roundStartMs = 0;

    // ===== TensorFlow.js ======================================================
    function loadTf() {
        if (window.tf) return Promise.resolve(window.tf);
        return new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
            s.onload = () => resolve(window.tf);
            s.onerror = () => reject(new Error("failed to load tfjs from CDN"));
            document.head.appendChild(s);
        });
    }

    // Rebuild the policy exactly like python/bonk/networks.py MLP:
    //   [Dense(no bias) -> LayerNorm(eps 1e-3) -> relu] x hidden -> Dense(+bias)
    // Weight record order: k0, ln0_g, ln0_b, k1, ln1_g, ln1_b, ..., kout, bout.
    // Architecture is inferred from the shapes, so any hidden size loads.
    function buildPolicy(weights) {
        const kernels = weights.filter((w) => w.shape.length === 2);
        const stateDim = kernels[0].shape[0];
        const actionDim = kernels[kernels.length - 1].shape[1];
        const hidden = kernels.slice(0, -1).map((k) => k.shape[1]);

        const input = tf.input({ shape: [stateDim] });
        let x = input;
        hidden.forEach((units, i) => {
            x = tf.layers.dense({ units, useBias: false, name: `pol_dense${i}` }).apply(x);
            x = tf.layers.layerNormalization({ epsilon: 1e-3, name: `pol_ln${i}` }).apply(x);
            x = tf.layers.activation({ activation: "relu", name: `pol_act${i}` }).apply(x);
        });
        const out = tf.layers.dense({ units: actionDim, name: "pol_out" }).apply(x);
        const model = tf.model({ inputs: input, outputs: out, name: "policy" });

        const tensors = weights.map((w) => tf.tensor(w.data, w.shape));
        model.setWeights(tensors);
        tensors.forEach((t) => t.dispose());
        console.log(`policy built: ${stateDim}->${hidden.join("->")}->${actionDim}`);
        if (stateDim !== STATE_DIM) console.warn(`expected obs dim ${STATE_DIM}, save says ${stateDim} — v1 model? use play2.mjs for those`);
        return model;
    }

    // One action index from the logits (argmax, or softmax sample).
    async function predict(obs) {
        const t = tf.tidy(() => policy.predict(tf.tensor2d([obs])));
        const logits = Array.from(await t.data());
        t.dispose();
        if (GREEDY) {
            let best = 0;
            for (let a = 1; a < logits.length; a++) if (logits[a] > logits[best]) best = a;
            return best;
        }
        const m = Math.max(...logits);
        const ps = logits.map((z) => Math.exp(z - m));
        let r = Math.random() * ps.reduce((a, b) => a + b, 0);
        for (let a = 0; a < ps.length; a++) { r -= ps[a]; if (r <= 0) return a; }
        return ps.length - 1;
    }

    // ===== Observation =========================================================
    function appliedKeys(id) {
        return top.GET_KEYS(keyMap.get(id) ?? 0);
    }
    function heavyAlpha(id) {
        let a = 0;
        try { a = top.playerids[id].playerData.children[HEAVY_CHILD].alpha; } catch (_) { a = 0; }
        return HEAVY_INVERT ? 1 - a : a;
    }
    function elapsedTicks() {
        if (typeof top.getCurrentFrame === "function" && roundStartFrame != null) {
            return Math.max(0, top.getCurrentFrame() - roundStartFrame);
        }
        return (performance.now() - roundStartMs) / (1000 / TPS);
    }
    // Heavy-meter memory (matches python/bonk2/env.py's tracker): while the
    // applied heavy key is held the meter is visible — record it; otherwise
    // report the last sighting and the (normalized) ticks since. Spawn = known
    // full, so a round starts at { seen: 1, seenAtTick: 0 }.
    function heavyFeatures(id, heavyHeld) {
        const now = elapsedTicks();
        let t = heavyTrack.get(id);
        if (!t) { t = { seen: 1, seenAtTick: 0 }; heavyTrack.set(id, t); }
        if (heavyHeld) {
            t.seen = heavyAlpha(id);
            t.seenAtTick = now;
        }
        const ticksSince = Math.max(0, now - t.seenAtTick);
        return [t.seen, Math.min(1, ticksSince / HEAVY_SEEN_TICKS_NORM)];
    }
    // The 12-number normalized block for one player (applied keys).
    function block(id) {
        const d = top.playerids[id].playerData2;
        const k = appliedKeys(id);
        const vs = velScale();
        const heavyValue = k.heavy ? heavyAlpha(id) : 0; // == heavyPower/1000
        const [seen, sinceNorm] = heavyFeatures(id, !!k.heavy);
        return [
            posX(d.px) * POS_SCALE, posY(d.py) * POS_SCALE,
            d.xvel * vs * VEL_SCALE, d.yvel * vs * VEL_SCALE,
            heavyValue,
            k.up ? 1 : 0, k.down ? 1 : 0, k.left ? 1 : 0, k.right ? 1 : 0, k.heavy ? 1 : 0,
            seen, sinceNorm,
        ];
    }
    function buildObs(meId, oppId) {
        const md = top.playerids[meId].playerData2;
        const od = top.playerids[oppId].playerData2;
        const vs = velScale();
        const ds = 1 / (top.scale * PPM); // position differences: offset cancels
        return block(meId).concat(block(oppId), [
            (od.px - md.px) * ds * POS_SCALE,
            (od.py - md.py) * ds * POS_SCALE,
            (od.xvel - md.xvel) * vs * VEL_SCALE,
            (od.yvel - md.yvel) * vs * VEL_SCALE,
            // pending block: the keys we last SENT (real network lag makes these
            // differ from our applied keys — exactly what the model trained on).
            lastSent.up, lastSent.down, lastSent.left, lastSent.right, lastSent.heavy,
            Math.min(1, elapsedTicks() / MAX_EPISODE_STEPS),
        ]);
    }

    // ===== Acting ==============================================================
    function playMove(actionIndex) {
        const keys = indexToKeys(actionIndex);
        top.presskeys(lastMove, keys);
        keyMap.set(top.myid, top.MAKE_KEYS(keys));
        lastMove = { ...keys };
        lastSent = {
            up: keys.up ? 1 : 0, down: keys.down ? 1 : 0,
            left: keys.left ? 1 : 0, right: keys.right ? 1 : 0,
            heavy: keys.heavy ? 1 : 0,
        };
        return keys;
    }

    // Opponent (and echoed self) key state from input packets.
    top.RECIEVEFUNCTION = function (args) {
        if (args.startsWith("42[7,")) {
            const data = JSON.parse(args.slice(2));
            keyMap.set(data[1], data[2].i);
        }
        if (args.startsWith("42[15,")) {
            top.presskeys(lastMove, { up: false, down: false, left: false, right: false, heavy: false, special: false });
            for (let i in top.playerids) {
                top.playerids[i].playerData2.alive = false;
            }
        }
        return args;
    };

    // ===== Player bookkeeping ==================================================
    function isAlive(id) {
        const p = top.playerids && top.playerids[id];
        return !!(p && p.playerData2 && p.playerData2.alive);
    }
    function getIds() {
        const me = top.myid;
        if (!isAlive(me)) return null;
        for (const pid in top.playerids) {
            if (pid != me && isAlive(pid)) return [me, parseInt(pid)];
        }
        return null;
    }

    // ===== Save-file loading (plain JSON or gzip) ==============================
    function loadFile() {
        return new Promise((resolve, reject) => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "application/json,.json,.gz";
            input.onchange = async (e) => {
                const f = e.target.files[0];
                if (!f) return;
                try {
                    const ds = f.stream().pipeThrough(new DecompressionStream("gzip"));
                    resolve(JSON.parse(await new Response(ds).text()));
                } catch (_) {
                    try { resolve(JSON.parse(await f.text())); }
                    catch (err) { reject(err); }
                }
            };
            input.click();
        });
    }

    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    // ===== Main loop ===========================================================
    async function main() {
        const save = await loadFile();
        const agent = save.agent || save;
        const records = agent.actor;   // bonk2 saves are PPO-only
        if (!records || !records.length) {
            throw new Error(`no actor weights in save (agent keys: ${Object.keys(agent)})`);
        }

        tf = await loadTf();
        await tf.ready();
        policy = buildPolicy(records);
        top.bonkai.policy = policy;
        console.log(
            `AI loaded [bonk2 actor] on tfjs (${tf.getBackend()})` +
            (save.progress ? `, episode ${save.progress.episodeCount}, ELO ${Math.round(save.progress.currentRating)}` : "") +
            `. Deciding every ${ACTION_REPEAT} frames (${GREEDY ? "greedy" : "sampled"}). ` +
            "Stop with top.bonkai.playStop().");

        let inRound = false;
        running = true;
        while (running) {
            const ids = getIds();
            if (ids) {
                if (!inRound) { // round just started: reset draw clock + trackers
                    inRound = true;
                    roundStartMs = performance.now();
                    roundStartFrame = typeof top.getCurrentFrame === "function"
                        ? top.getCurrentFrame() : null;
                    heavyTrack.clear(); // spawn = heavy known full
                }
                const [me, opp] = ids;
                let actionIndex = 0;
                try {
                    actionIndex = await predict(buildObs(me, opp));
                } catch (err) {
                    console.error("inference error:", err);
                }
                playMove(actionIndex);
            } else if (inRound) { // round ended: clear per-round state
                inRound = false;
                top.presskeys(lastMove, { up: false, down: false, left: false, right: false, heavy: false, special: false });
                lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
                lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 };
                heavyTrack.clear();
            }
            await sleep(DECIDE_MS);
        }
        console.log("AI stopped.");
    }

    top.bonkai.playStop = () => { running = false; };
    top.bonkai.playRestart = () => { if (!running) main().catch((e) => console.error(e)); };

    main().catch((e) => console.error("play3.mjs failed:", e));
})();
