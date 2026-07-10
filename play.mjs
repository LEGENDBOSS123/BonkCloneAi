// play.mjs — paste-into-the-tab AI player for the PPO model trained by this
// project. It prompts for a save file (the JSON produced by "Save progress" in
// train.html), rebuilds the actor as a TensorFlow.js model (loaded from CDN) and
// runs inference through it, reads the live bonk.io game state, and drives your
// player.
//
// REQUIREMENTS
//   1. The bonk.io instrumentation must already be injected in the tab — the same
//      `script.js` you use from the Bonk1v1Ai project. It provides the hooks used
//      below: top.playerids, top.myid, top.scale, top.presskeys, top.MAKE_KEYS,
//      top.GET_KEYS and the top.RECIEVEFUNCTION websocket hook point.
//   2. Network access (it loads @tensorflow/tfjs from jsDelivr on first run).
//   3. Paste this whole file into the console AFTER that. It will pop a file
//      picker; choose your saved model JSON and it starts playing. Stop with
//      top.bonkai.playStop().
//
// OBSERVATION must match src/env.mjs exactly:
//   per player: [x, y, vx, vy, heavyValue, up, down, left, right, heavy]
//   full state (from "me" POV): self(10) + opponent(10) + relative[dx,dy,dvx,dvy]
//
// NOTE ON SIM-TO-REAL: the model learned this project's Planck sim + custom heavy
// mechanic, which are only approximations of real bonk.io physics. Transfer is
// imperfect; the scale constants below and the heavy dynamics are the first things
// to tune if it plays oddly.

(function () {
    "use strict";
    top.bonkai = top.bonkai || {};

    // ===== Tunables (match src/config.mjs / src/env.mjs) ===================
    // COORDINATE CONVERSION (derived from the Bonk1v1Ai project):
    //   bonk stores live positions as  px = mapCoord * top.scale  (top.scale is
    //   bonk's own physics scale, captured by the instrumentation). So the map
    //   coordinate is px / top.scale. Our env then works in mapCoord / ppm, with
    //   ppm = 15 from map1.json. Therefore:  ourPos = (px / top.scale) / PPM.
    const PPM = 15;                    // map1.json physics.ppm
    const TPS = 30;                    // our env ticks/second (map.mjs TPS)
    // If bonk's xvel/yvel are per-FRAME (the usual case), converting to our
    // per-SECOND (Planck m/s) velocities needs a *TPS factor. Set VEL_TIME = 1 if
    // they turn out to be per-second (velocities feel ~30x too small/large).
    const VEL_TIME = 1000;
    top.VEL_TIME = VEL_TIME;
    // COORDINATE ORIGIN: bonk's live px/py have their origin at a map corner, but
    // map1.json (and this project's env) are centered at 0,0. So subtract the live
    // map-center in map-coords before scaling. Measured from spawns: right/left
    // spawn map-coords were 465/265 (center 365, ±100 = the spawns), y-center 250.
    // Recalibrate for a different map: spawn on both sides, ORIGIN_X = average of
    // (px/top.scale); ORIGIN_Y = (py/top.scale) - spawnY (spawnY = -25 here).
    const ORIGIN_X = 365, ORIGIN_Y = 250; // live map-coords of the map center
    // Heavy: our env's heavyValue was heavyPower in [0,1000]. Bonk exposes the
    // heavy charge as a sprite alpha in [0,1] (playerData.children[HEAVY_CHILD].alpha,
    // same as Bonk1v1Ai). Assuming they're linearly related, heavyPower ≈ alpha*1000.
    const HEAVY_MAX = 1000;            // scale alpha (0..1) -> our heavyPower range
    const HEAVY_CHILD = 2;             // sprite child index holding the heavy alpha (bonk-version specific)
    const HEAVY_INVERT = false;        // set true if a FULL charge reads as alpha 0 instead of 1
    const ACTIVATION = "tanh";         // config.network.activation ("tanh" or "relu")
    const DECIDE_MS = 1000 / 30;       // decision cadence (~30 fps)

    // Live px -> our units. Absolute positions are re-centered (subtract ORIGIN);
    // velocities and coordinate DIFFERENCES are offset-free (offset cancels).
    const posX = (px) => (px / top.scale - ORIGIN_X) / PPM;
    const posY = (py) => (py / top.scale - ORIGIN_Y) / PPM;
    const velScale = () => top.VEL_TIME / (top.scale * PPM); // no offset for velocity/deltas

    // ===== State kept locally =============================================
    const keyMap = new Map();          // playerId -> encoded key int (from packets / our own)
    const lastKeys = new Map();        // playerId -> {up,down,left,right,heavy}
    let tf = null;                     // TensorFlow.js (loaded from CDN)
    let actor = null;                  // the tf.LayersModel
    let running = false;
    let lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };

    // ===== TensorFlow.js =================================================
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

    // Rebuild the actor exactly like src/rl/network.mjs so the saved weights load
    // in order. Architecture is inferred from the weight shapes (2D = Dense kernel):
    //   stateDim = first kernel rows, actionDim = last kernel cols, hidden = the
    //   rest. LayerNorm is present iff there are 3 tensors per hidden block (kernel
    //   + gamma + beta) rather than 2 (kernel + bias).
    function buildActor(weights) {
        const kernels = weights.filter((w) => w.shape.length === 2);
        const stateDim = kernels[0].shape[0];
        const actionDim = kernels[kernels.length - 1].shape[1];
        const hidden = kernels.slice(0, -1).map((k) => k.shape[1]);
        const useLayerNorm = weights.length === 3 * hidden.length + 2;

        const input = tf.input({ shape: [stateDim] });
        let x = input;
        hidden.forEach((units, i) => {
            x = tf.layers.dense({ units, useBias: !useLayerNorm, name: `actor_dense${i}` }).apply(x);
            if (useLayerNorm) x = tf.layers.layerNormalization({ name: `actor_ln${i}` }).apply(x);
            x = tf.layers.activation({ activation: ACTIVATION, name: `actor_act${i}` }).apply(x);
        });
        const out = tf.layers.dense({ units: actionDim, name: "actor_out" }).apply(x);
        const model = tf.model({ inputs: input, outputs: out, name: "actor" });

        const tensors = weights.map((w) => tf.tensor(w.data, w.shape));
        model.setWeights(tensors);
        tensors.forEach((t) => t.dispose());
        console.log(`actor built: ${stateDim}->${hidden.join("->")}->${actionDim}, layerNorm=${useLayerNorm}`);
        return model;
    }

    // Greedy multi-binary action from the policy: sigmoid(logit) > 0.5  <=>  logit > 0.
    async function predict(obs) {
        const t = tf.tidy(() => actor.predict(tf.tensor2d([obs])).greater(0).cast("int32"));
        const arr = await t.data();
        t.dispose();
        return Array.from(arr);
    }

    // ===== Observation (must mirror src/env.mjs) =========================
    function keysOf(id) {
        return lastKeys.get(id) || top.GET_KEYS(keyMap.get(id) ?? 0);
    }
    // Live heavy charge (bonk sprite alpha, 0..1) scaled to our heavyPower range.
    function heavyPower(id) {
        let a = 0;
        try { a = top.playerids[id].playerData.children[HEAVY_CHILD].alpha; } catch (_) { a = 0; }
        if (HEAVY_INVERT) a = 1 - a;
        return a * HEAVY_MAX;
    }
    function features(id) {
        const d = top.playerids[id].playerData2;
        const k = keysOf(id);
        const vs = velScale();
        // env.mjs: heavyValue = heavyHeld ? heavyPower : 0.
        const heavyValue = k.heavy ? heavyPower(id) : 0;
        return [
            posX(d.px), posY(d.py), d.xvel * vs, d.yvel * vs, heavyValue,
            k.up ? 1 : 0, k.down ? 1 : 0, k.left ? 1 : 0, k.right ? 1 : 0, k.heavy ? 1 : 0,
        ];
    }
    function buildObs(meId, oppId) {
        const md = top.playerids[meId].playerData2, od = top.playerids[oppId].playerData2;
        const vs = velScale();
        const ds = 1 / (top.scale * PPM); // scale for position DIFFERENCES (offset cancels)
        const x = features(meId).concat(features(oppId), [
            (od.px - md.px) * ds,
            (od.py - md.py) * ds,
            (od.xvel - md.xvel) * vs,
            (od.yvel - md.yvel) * vs,
        ]);
        console.log(x);
        return x;
    }

    // ===== Acting ========================================================
    // action is [up, down, left, right, heavy] (0/1) — env.mjs order.
    function playMove(action) {
        const keys = {
            up: !!action[0], down: !!action[1], left: !!action[2], right: !!action[3],
            heavy: !!action[4], special: false,
        };
        top.presskeys(lastMove, keys);
        keyMap.set(top.myid, top.MAKE_KEYS(keys));
        lastMove = { ...keys };
        return keys;
    }

    // Keep the opponent's key state synced from their input packets, for the obs.
    top.RECIEVEFUNCTION = function (args) {
        if (args.startsWith("42[7,")) {
            const data = JSON.parse(args.slice(2));
            keyMap.set(data[1], data[2].i);
        }
        return args;
    };

    // ===== Player bookkeeping ============================================
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

    // ===== Save-file loading (plain JSON or gzip) ========================
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

    // ===== Main loop =====================================================
    async function main() {
        const save = await loadFile();
        if (!save.actor || !save.actor.length) throw new Error("save file has no `actor` weights");

        tf = await loadTf();
        await tf.ready();
        actor = buildActor(save.actor);
        top.bonkai.actor = actor;
        console.log(
            `AI loaded on tfjs (${tf.getBackend()})` +
            (save.progress ? `, episode ${save.progress.episodeCount}, ELO ${Math.round(save.progress.currentRating)}` : "") +
            ". Playing. Stop with top.bonkai.playStop().");

        running = true;
        while (running) {
            const ids = getIds();
            if (ids) {
                const [me, opp] = ids;

                // Track the opponent's held keys (for their obs block); heavy value
                // now comes from the live sprite alpha, so no counter to advance.
                const ok = top.GET_KEYS(keyMap.get(opp) ?? 0);
                lastKeys.set(opp, { up: !!ok.up, down: !!ok.down, left: !!ok.left, right: !!ok.right, heavy: !!ok.heavy });

                let action;
                try {
                    action = await predict(buildObs(me, opp)); // [up,down,left,right,heavy] 0/1
                } catch (err) {
                    console.error("inference error:", err);
                    action = [0, 0, 0, 0, 0];
                }

                const keys = playMove(action);
                lastKeys.set(me, { up: keys.up, down: keys.down, left: keys.left, right: keys.right, heavy: keys.heavy });
            } else {
                // Between rounds: reset per-episode tracking.
                lastKeys.clear();
                lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
            }
            await sleep(DECIDE_MS);
        }
        console.log("AI stopped.");
    }

    top.bonkai.playStop = () => { running = false; };
    top.bonkai.playRestart = () => { if (!running) main().catch((e) => console.error(e)); };

    main().catch((e) => console.error("play.mjs failed:", e));
})();
