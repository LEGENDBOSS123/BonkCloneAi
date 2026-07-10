// play2.mjs — paste-into-the-tab AI player for the rl2 / PyTorch checkpoints
// (python -m bonk.train_ppo / train_ppo_mp / train). Prompts for the checkpoint
// JSON, rebuilds the policy net with TensorFlow.js, reads live bonk.io state,
// and drives your player.
//
// REQUIREMENTS (same as play.mjs)
//   1. The bonk.io instrumentation from Bonk1v1Ai must already be injected:
//      top.playerids, top.myid, top.scale, top.presskeys, top.MAKE_KEYS,
//      top.GET_KEYS, top.RECIEVEFUNCTION (and ideally top.getCurrentFrame).
//   2. Network access (loads @tensorflow/tfjs from jsDelivr once).
//   3. Paste AFTER the instrumentation; pick the checkpoint in the file dialog.
//      Stop with top.bonkai.playStop().
//
// Works with BOTH algorithms: PPO saves (agent.actor) and NFSP saves
// (agent.avg — the deployable average policy). Auto-detected.
//
// OBSERVATION must match python/bonk/env2.py (and src/rl2/lag_env.mjs), 30 dims:
//   [ 0- 9] self : x,y,vx,vy (normalized), heavyValue, up,down,left,right,heavy
//   [10-19] opp  : same, with the opponent's APPLIED keys
//   [20-23] rel  : dx,dy,dvx,dvy (normalized)
//   [24-28] pend : the keys WE LAST SENT (may not have applied yet — real lag!)
//   [29]    time : ticks since round start / MAX_EPISODE_STEPS
// Policy outputs 18 logits over joint actions: index = lr*6 + ud*2 + heavy.

(function () {
    "use strict";
    top.bonkai = top.bonkai || {};

    // ===== Coordinate conversion (calibrated in play.mjs — see its notes) ====
    const PPM = 10;                    // map1.json physics.ppm
    const ORIGIN_X = 365, ORIGIN_Y = 250; // live map-coords of the map center
    // bonk's playerData2.xvel is a finite difference in position-units per
    // MILLISECOND (the instrumentation divides by performance.now() deltas), so
    // m/s = xvel * 1000 / (top.scale * PPM). 900 was the old empirical value —
    // fall back to it if movement reads look off.
    const VEL_TIME = 1000;
    top.VEL_TIME = VEL_TIME;

    // ===== rl2 env constants (python/bonk/config.py) ==========================
    const POS_SCALE = 1 / 30;
    const VEL_SCALE = 1 / 30;
    const MAX_EPISODE_STEPS = 2500;    // draw-clock horizon (ticks)
    const ACTION_REPEAT = 4;           // decide every 4 frames, hold between
    const TPS = 30;
    const DECIDE_MS = (1000 / TPS) * ACTION_REPEAT;
    const GREEDY = true;               // argmax (set false to sample the policy)

    // Heavy charge: bonk sprite alpha in [0,1]; our obs heavyValue is
    // heavyPower/1000 = exactly that alpha.
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
    let tf = null;
    let policy = null;
    let running = false;
    let lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
    let lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 }; // pending block
    let roundStartFrame = null;        // for the draw-clock feature
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
        if (stateDim !== 30) console.warn(`expected obs dim 30, save says ${stateDim} — obs code may be stale`);
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
    // The 10-number normalized block for one player (applied keys).
    function block(id) {
        const d = top.playerids[id].playerData2;
        const k = appliedKeys(id);
        const vs = velScale();
        const heavyValue = k.heavy ? heavyAlpha(id) : 0; // == heavyPower/1000
        return [
            posX(d.px) * POS_SCALE, posY(d.py) * POS_SCALE,
            d.xvel * vs * VEL_SCALE, d.yvel * vs * VEL_SCALE,
            heavyValue,
            k.up ? 1 : 0, k.down ? 1 : 0, k.left ? 1 : 0, k.right ? 1 : 0, k.heavy ? 1 : 0,
        ];
    }
    function elapsedTicks() {
        if (typeof top.getCurrentFrame === "function" && roundStartFrame != null) {
            return Math.max(0, top.getCurrentFrame() - roundStartFrame);
        }
        return (performance.now() - roundStartMs) / (1000 / TPS);
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
        if(args.startsWith("42[15,")){
            top.presskeys(lastMove, { up: false, down: false, left: false, right: false, heavy: false, special: false });
            for(let i in top.playerids){
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
        // PPO -> actor; NFSP -> avg (the deployable average policy).
        const records = agent.actor || agent.avg;
        if (!records || !records.length) {
            throw new Error(`no actor/avg weights in save (agent keys: ${Object.keys(agent)})`);
        }
        const which = agent.actor ? "actor (PPO)" : "avg (NFSP)";

        tf = await loadTf();
        await tf.ready();
        policy = buildPolicy(records);
        top.bonkai.policy = policy;
        console.log(
            `AI loaded [${which}] on tfjs (${tf.getBackend()})` +
            (save.progress ? `, episode ${save.progress.episodeCount}, ELO ${Math.round(save.progress.currentRating)}` : "") +
            `. Deciding every ${ACTION_REPEAT} frames (${GREEDY ? "greedy" : "sampled"}). ` +
            "Stop with top.bonkai.playStop().");

        let inRound = false;
        running = true;
        while (running) {
            const ids = getIds();
            if (ids) {
                if (!inRound) { // round just started: reset the draw clock
                    inRound = true;
                    roundStartMs = performance.now();
                    roundStartFrame = typeof top.getCurrentFrame === "function"
                        ? top.getCurrentFrame() : null;
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
                
            }
            await sleep(DECIDE_MS);
        }
        console.log("AI stopped.");
    }

    top.bonkai.playStop = () => { running = false; };
    top.bonkai.playRestart = () => { if (!running) main().catch((e) => console.error(e)); };

    main().catch((e) => console.error("play2.mjs failed:", e));
})();
