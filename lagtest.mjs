// lagtest.mjs — measure the REAL input lag of bonk.io, so bonk2/config.py's
// SELF_LAG / VIEW_LAG stop being guesses.
//
// REQUIREMENTS
//   1. The usual instrumentation injected: top.playerids, top.myid,
//      top.presskeys, top.MAKE_KEYS, top.GET_KEYS, top.RECIEVEFUNCTION, and
//      top.getCurrentFrame (this script is useless without the frame counter).
//   2. Be ALIVE in a round, standing still, with room to move sideways.
//   3. STOP play3.mjs first (top.bonkai.playStop()) — both scripts drive your
//      keys and hook RECIEVEFUNCTION, so they will fight each other.
//
// WHAT YOU GET
//   top.lagtest.send(action)   -> presses that action, returns the frame it was
//                                 sent on. Manual mode: note the number, watch
//                                 when your ball actually moves, subtract.
//   top.lagtest.self()         -> ONE automatic measurement: sends a move, then
//                                 polls every frame until your own velocity
//                                 actually changes. Returns the frame delta.
//   top.lagtest.selfRepeat(n)  -> n measurements + stats. USE THIS ONE. A single
//                                 sample is worthless: where the keypress lands
//                                 inside a frame quantizes the answer, so you
//                                 need the distribution, not one number.
//   top.lagtest.echo(n)        -> round-trip: frames from sending your keys to
//                                 the server echoing them back to you.
//   top.lagtest.stop()         -> release keys, unhook, restore.
//
// Action indices are the training encoding (index = lr*6 + ud*2 + heavy):
//   0 = idle, 12 = right, 6 = left, 2 = up, 1 = heavy.

(function () {
    "use strict";
    top.lagtest = top.lagtest || {};

    const TPS = 30;
    const IDLE = 0, RIGHT = 12, LEFT = 6, UP = 2;

    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const frame = () =>
        (typeof top.getCurrentFrame === "function" ? top.getCurrentFrame() : null);

    function indexToKeys(index) {
        const lr = (index / 6) | 0;
        const ud = ((index / 2) | 0) % 3;
        return {
            left: lr === 1, right: lr === 2,
            up: ud === 1, down: ud === 2,
            heavy: index % 2 === 1, special: false,
        };
    }

    let lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };

    // Raw key press, no model. Returns the frame it went out on.
    function sendKeys(actionIndex) {
        const keys = indexToKeys(actionIndex);
        const f = frame();
        top.presskeys(lastMove, keys);
        lastMove = { ...keys };
        return f;
    }

    // ===== Live observation (mirrors play3.mjs / bonk2 env, 34-dim) ==========
    // Built so send() feeds the model the SAME thing it sees at deploy, and the
    // action it returns is a real decision rather than a forward on zeros.
    const PPM = 15, ORIGIN_X = 365, ORIGIN_Y = 250, VEL_TIME = 1000;
    const POS_SCALE = 1 / 30, VEL_SCALE = 1 / 30;
    const HEAVY_SEEN_TICKS_NORM = 200, MAX_EPISODE_STEPS = 3000;
    const HEAVY_CHILD = 2, HEAVY_INVERT = false;

    const posX = (px) => (px / top.scale - ORIGIN_X) / PPM;
    const posY = (py) => (py / top.scale - ORIGIN_Y) / PPM;
    const velScale = () => VEL_TIME / (top.scale * PPM);

    // Opponent applied keys arrive as input packets; hook to track them.
    const keyMap = new Map();
    const heavyTrack = new Map();
    let lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 };
    let keyHooked = false, prevKeyRecv = null;

    function hookKeys() {
        if (keyHooked) return;
        prevKeyRecv = top.RECIEVEFUNCTION;
        keyHooked = true;
        top.RECIEVEFUNCTION = function (args) {
            try {
                if (args.startsWith("42[7,")) {
                    const d = JSON.parse(args.slice(2));
                    keyMap.set(d[1], d[2].i);
                }
            } catch (_) { /* never break the socket */ }
            return prevKeyRecv ? prevKeyRecv(args) : args;
        };
    }

    const appliedKeys = (id) => top.GET_KEYS(keyMap.get(id) ?? 0);
    function heavyAlpha(id) {
        let a = 0;
        try { a = top.playerids[id].playerData.children[HEAVY_CHILD].alpha; } catch (_) { a = 0; }
        return HEAVY_INVERT ? 1 - a : a;
    }
    function heavyFeatures(id, held, nowTick) {
        let t = heavyTrack.get(id);
        if (!t) { t = { seen: 1, at: nowTick }; heavyTrack.set(id, t); }
        if (held) { t.seen = heavyAlpha(id); t.at = nowTick; }
        return [t.seen, Math.min(1, Math.max(0, nowTick - t.at) / HEAVY_SEEN_TICKS_NORM)];
    }
    function block(id, nowTick) {
        const d = top.playerids[id].playerData2;
        const k = appliedKeys(id);
        const vs = velScale();
        const [seen, since] = heavyFeatures(id, !!k.heavy, nowTick);
        return [
            posX(d.px) * POS_SCALE, posY(d.py) * POS_SCALE,
            d.xvel * vs * VEL_SCALE, d.yvel * vs * VEL_SCALE,
            k.heavy ? heavyAlpha(id) : 0,
            k.up ? 1 : 0, k.down ? 1 : 0, k.left ? 1 : 0, k.right ? 1 : 0, k.heavy ? 1 : 0,
            seen, since,
        ];
    }
    function buildObs(meId, oppId, nowTick) {
        const md = top.playerids[meId].playerData2, od = top.playerids[oppId].playerData2;
        const vs = velScale(), ds = 1 / (top.scale * PPM);
        return block(meId, nowTick).concat(block(oppId, nowTick), [
            (od.px - md.px) * ds * POS_SCALE, (od.py - md.py) * ds * POS_SCALE,
            (od.xvel - md.xvel) * vs * VEL_SCALE, (od.yvel - md.yvel) * vs * VEL_SCALE,
            lastSent.up, lastSent.down, lastSent.left, lastSent.right, lastSent.heavy,
            Math.min(1, nowTick / MAX_EPISODE_STEPS),
        ]);
    }
    function opponentId() {
        for (const pid in top.playerids) {
            if (pid != top.myid) {
                const p = top.playerids[pid];
                if (p && p.playerData2 && p.playerData2.alive) return parseInt(pid);
            }
        }
        return null;
    }

    // ===== Helpers ============================================================
    function me() { return top.playerids[top.myid]; }
    function myData() { return me().playerData2; }
    function alive() {
        const p = me();
        return !!(p && p.playerData2 && p.playerData2.alive);
    }

    // Block until the frame counter ticks over.
    async function nextFrame() {
        const f0 = frame();
        let guard = 0;
        while (frame() === f0 && guard++ < 2000) await sleep(1);
        return frame();
    }
    async function waitFrames(n) {
        for (let i = 0; i < n; i++) await nextFrame();
    }

    function stats(xs) {
        if (!xs.length) return null;
        const s = [...xs].sort((a, b) => a - b);
        const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
        return {
            n: xs.length,
            min: s[0],
            median: s[(s.length / 2) | 0],
            max: s[s.length - 1],
            mean: +mean.toFixed(2),
            meanMs: +(mean * 1000 / TPS).toFixed(1),
        };
    }

    // ===== Detection ==========================================================
    // Detect on POSITION, not velocity. playerData2.xvel is a finite difference
    // the instrumentation computes from px over performance.now() deltas, so it
    // lags, jitters, and needs a big threshold to clear the noise — at
    // MOVE_ACCEL=12 m/s^2 a 1.3 m/s threshold alone costs ~3.5 frames. px moves
    // on the very first frame the input applies.
    //
    // And the ball never stops on its own (friction 0, no damping), so every
    // sample must start from a VERIFIED standstill or the baseline is garbage.
    async function waitUntilStill(maxFrames = 150, tol = 0.02) {
        let stable = 0, prev = myData().px;
        for (let i = 0; i < maxFrames; i++) {
            await nextFrame();
            const p = myData().px;
            if (Math.abs(p - prev) < tol) stable++; else stable = 0;
            prev = p;
            if (stable >= 6) return true;
        }
        return false;   // still drifting — sample will be skipped
    }

    // Measure the stationary px noise floor so the trigger threshold is
    // derived, not guessed.
    async function noiseFloor(frames = 20) {
        const p0 = myData().px;
        let worst = 0;
        for (let i = 0; i < frames; i++) {
            await nextFrame();
            worst = Math.max(worst, Math.abs(myData().px - p0));
        }
        return worst;
    }

    // ===== RAW TRACE — run this FIRST, trust it over any summary ==============
    // Prints px/xvel per frame around a keypress so you can see with your own
    // eyes which frame motion starts on, and whether px even updates per frame.
    async function trace(action = RIGHT, after = 14) {
        if (!alive()) { console.warn("not alive"); return null; }
        sendKeys(IDLE);
        const still = await waitUntilStill();
        if (!still) console.warn("ball never settled — trace may be misleading");
        const p0 = myData().px, f0 = frame();
        const rows = [{ frame: 0, px: 0, dpx: 0, xvel: +myData().xvel.toFixed(4), note: "baseline" }];
        const sent = sendKeys(action);
        let prev = p0;
        for (let i = 0; i < after; i++) {
            const f = await nextFrame();
            const p = myData().px;
            rows.push({
                frame: f - sent,
                px: +(p - p0).toFixed(4),
                dpx: +(p - prev).toFixed(4),
                xvel: +myData().xvel.toFixed(4),
                note: Math.abs(p - p0) > 0.02 && rows.every(r => r.note !== "MOVING") ? "MOVING" : "",
            });
            prev = p;
        }
        sendKeys(IDLE);
        console.log(`sent on frame ${sent} (baseline frame ${f0}); "frame" column is relative to send`);
        console.table(rows);
        console.log(
            "Read the first row marked MOVING (or where dpx first goes non-zero):\n" +
            "  that relative frame number IS your input lag.\n" +
            "If dpx is zero for several frames then jumps, px is only updating on\n" +
            "network snapshots rather than per frame — tell me and we measure differently.");
        return rows;
    }

    // ===== SELF_LAG: keypress -> your own position actually changing ==========
    async function selfOnce(opts = {}) {
        const { action = RIGHT, timeout = 40 } = opts;
        if (!alive()) { console.warn("not alive — get in a round first"); return null; }

        sendKeys(IDLE);
        if (!await waitUntilStill()) return null;   // never settled; skip sample
        const tol = Math.max(0.02, (await noiseFloor(10)) * 2.5);

        const p0 = myData().px;
        const sent = sendKeys(action);
        let effect = null;
        for (let i = 0; i < timeout; i++) {
            const f = await nextFrame();
            if (Math.abs(myData().px - p0) > tol) { effect = f; break; }
        }
        sendKeys(IDLE);
        if (effect == null) {
            console.warn("no motion detected — stuck against a wall?");
            return null;
        }
        return {
            sentFrame: sent,
            effectFrame: effect,
            lagFrames: effect - sent,
            lagMs: +((effect - sent) * 1000 / TPS).toFixed(1),
        };
    }

    async function selfRepeat(n = 15, opts = {}) {
        const out = [];
        // Alternate directions so you don't drift off the map over 15 runs.
        for (let i = 0; i < n; i++) {
            const r = await selfOnce({ ...opts, action: i % 2 ? LEFT : RIGHT });
            if (r) out.push(r.lagFrames);
            await waitFrames(10);
            if (!alive()) { console.warn("died mid-test; stopping early"); break; }
        }
        const st = stats(out);
        console.log("raw lagFrames:", out.join(", "));
        console.table([st]);
        if (st) {
            console.log(
                `\n=> SELF_LAG measured: median ${st.median} frames ` +
                `(${(st.median * 1000 / TPS).toFixed(0)} ms), range ${st.min}-${st.max}.\n` +
                `   Put this in python/bonk2/config.py as:\n` +
                `     SELF_LAG_MIN = ${st.min}\n` +
                `     SELF_LAG_MAX = ${st.max}\n` +
                `   NOTE: this includes ~1 frame of measurement floor — the effect\n` +
                `   can only be observed on the frame AFTER it is applied.`);
        }
        return st;
    }

    // ===== Round-trip: your keys -> server -> echoed back to you ==============
    // Useful for estimating VIEW_LAG: the opponent's input reaches you over a
    // comparable path, so their staleness is roughly this round-trip (their
    // upload + your download). Rough, but grounded in a real measurement
    // instead of a guess.
    let echoHooked = false, prevRecv = null, pendingEcho = null;
    const echoSamples = [];

    function hookEcho() {
        if (echoHooked) return;
        prevRecv = top.RECIEVEFUNCTION;
        echoHooked = true;
        top.RECIEVEFUNCTION = function (args) {
            try {
                if (pendingEcho != null && args.startsWith("42[7,")) {
                    const data = JSON.parse(args.slice(2));
                    if (data[1] == top.myid) {
                        echoSamples.push(frame() - pendingEcho);
                        pendingEcho = null;
                    }
                }
            } catch (_) { /* never break the socket */ }
            return prevRecv ? prevRecv(args) : args;
        };
    }

    async function echo(n = 15) {
        hookEcho();
        echoSamples.length = 0;
        for (let i = 0; i < n; i++) {
            pendingEcho = frame();
            sendKeys(i % 2 ? LEFT : RIGHT);
            await waitFrames(12);
            sendKeys(IDLE);
            await waitFrames(6);
        }
        const st = stats(echoSamples);
        console.log("raw echo frames:", echoSamples.join(", "));
        console.table([st]);
        if (st) {
            console.log(
                `\n=> Round-trip: median ${st.median} frames ` +
                `(${(st.median * 1000 / TPS).toFixed(0)} ms).\n` +
                `   The opponent's inputs travel a comparable path, so VIEW_LAG is\n` +
                `   in this ballpark. Widen the config range around it:\n` +
                `     VIEW_LAG_MIN = ${Math.max(0, st.min - 1)}\n` +
                `     VIEW_LAG_MAX = ${st.max + 2}\n` +
                `   (Assumes roughly symmetric up/down latency — treat as an estimate.)`);
        }
        return st;
    }

    // ===== MODEL DELAY ========================================================
    // The agent's real self-lag is NOT just the game's input lag: every decision
    // costs an obs build + a tfjs forward (which ends in an await'd GPU->CPU
    // sync) before the keys are even sent. That time is added on top. Measure
    // it, then compare against selfRepeat() to isolate the model's share.
    let tf = null, policy = null, stateDim = null;

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

    function buildPolicy(weights) {
        const kernels = weights.filter((w) => w.shape.length === 2);
        stateDim = kernels[0].shape[0];
        const actionDim = kernels[kernels.length - 1].shape[1];
        const hidden = kernels.slice(0, -1).map((k) => k.shape[1]);
        const input = tf.input({ shape: [stateDim] });
        let x = input;
        hidden.forEach((units, i) => {
            x = tf.layers.dense({ units, useBias: false, name: `d${i}` }).apply(x);
            x = tf.layers.layerNormalization({ epsilon: 1e-3, name: `ln${i}` }).apply(x);
            x = tf.layers.activation({ activation: "relu", name: `a${i}` }).apply(x);
        });
        const out = tf.layers.dense({ units: actionDim, name: "out" }).apply(x);
        const model = tf.model({ inputs: input, outputs: out });
        const tensors = weights.map((w) => tf.tensor(w.data, w.shape));
        model.setWeights(tensors);
        tensors.forEach((t) => t.dispose());
        console.log(`policy built: ${stateDim}->${hidden.join("->")}->${actionDim}`);
        return model;
    }

    function pickFile() {
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

    // Pick an EXPORTED slim model (models/bonk2-ppo-latest.json). A raw
    // training checkpoint is ~900 MB and JSON.parse will blow up in the browser.
    async function loadModel() {
        const save = await pickFile();
        const agent = save.agent || save;
        const records = agent.actor;
        if (!records || !records.length) throw new Error("no actor weights in that file");
        tf = await loadTf();
        await tf.ready();
        const best = await benchBackends(records);   // also warms up
        policy = best.model;
        console.log(
            `model ready on '${tf.getBackend()}'.\n` +
            `If this backend differs from webgl, play3.mjs is leaving that speedup\n` +
            `on the table — add tf.setBackend('${tf.getBackend()}') before building\n` +
            `its policy, and swap its "await t.data()" for "t.dataSync()".`);
        return true;
    }

    // dataSync(), not await data(). On WebGL the async readback has to queue
    // behind the game's own draw calls, and awaiting also yields the event loop
    // so unrelated work lands inside the measurement. For a ~79k-MAC net the
    // blocking read is sub-millisecond on cpu/wasm.
    async function predict(obs) {
        const t = tf.tidy(() => policy.predict(tf.tensor2d([obs])));
        const logits = t.dataSync();
        t.dispose();
        let best = 0;
        for (let a = 1; a < logits.length; a++) if (logits[a] > logits[best]) best = a;
        return best;
    }

    // Small MLPs are usually FASTER on cpu/wasm than webgl: the matmuls are
    // trivial, so the GPU upload + readback dominates — and here it also
    // contends with Pixi rendering the game. Rebuild on each backend and time
    // it rather than assuming.
    async function benchBackends(records, n = 40) {
        const results = [];
        const want = ["cpu", "wasm", "webgl"];
        for (const b of want) {
            try {
                if (!(await tf.setBackend(b))) continue;
                await tf.ready();
                const m = buildPolicy(records);
                const obs = new Array(stateDim).fill(0);
                for (let i = 0; i < 10; i++) {           // warm up / compile
                    const t = tf.tidy(() => m.predict(tf.tensor2d([obs]))); t.dataSync(); t.dispose();
                }
                const t0 = performance.now();
                for (let i = 0; i < n; i++) {
                    const t = tf.tidy(() => m.predict(tf.tensor2d([obs]))); t.dataSync(); t.dispose();
                }
                const ms = (performance.now() - t0) / n;
                results.push({ backend: b, ms: +ms.toFixed(3), frames: +(ms * TPS / 1000).toFixed(3), model: m });
            } catch (e) {
                console.log(`backend ${b}: unavailable (${e.message})`);
            }
        }
        if (!results.length) throw new Error("no tfjs backend worked");
        results.sort((a, b) => a.ms - b.ms);
        console.table(results.map(({ model, ...r }) => r));
        const best = results[0];
        await tf.setBackend(best.backend);
        await tf.ready();
        console.log(`=> fastest backend: ${best.backend} at ${best.ms.toFixed(2)} ms ` +
                    `(${best.frames.toFixed(2)} frames per decision)`);
        return best;
    }

    // Pure inference cost, no game interaction needed.
    async function infer(n = 50) {
        if (!policy) { console.warn("call await top.lagtest.loadModel() first"); return null; }
        const obs = new Array(stateDim).fill(0);
        const ms = [];
        for (let i = 0; i < n; i++) {
            const t0 = performance.now();
            await predict(obs);
            ms.push(performance.now() - t0);
        }
        const st = stats(ms);
        console.table([{ ...st, frames: +(st.mean * TPS / 1000).toFixed(2) }]);
        console.log(
            `\n=> inference: mean ${st.mean.toFixed(1)} ms ` +
            `= ${(st.mean * TPS / 1000).toFixed(2)} frames of pure model delay,\n` +
            `   on top of whatever the game's own input lag is.`);
        return st;
    }

    // FULL pipeline: run a real forward (paying its real cost), then send a
    // known action and watch for the effect. Total = model + send + game lag.
    async function pipelineOnce(opts = {}) {
        const { action = RIGHT, timeout = 40 } = opts;
        if (!policy) { console.warn("call await top.lagtest.loadModel() first"); return null; }
        if (!alive()) { console.warn("not alive"); return null; }
        sendKeys(IDLE);
        if (!await waitUntilStill()) return null;
        const tol = Math.max(0.02, (await noiseFloor(10)) * 2.5);

        const p0 = myData().px;
        const start = frame();                       // decision cycle begins
        await predict(new Array(stateDim).fill(0));  // pay the real inference cost
        sendKeys(action);                                // ...then act
        let effect = null;
        for (let i = 0; i < timeout; i++) {
            const f = await nextFrame();
            if (Math.abs(myData().px - p0) > tol) { effect = f; break; }
        }
        sendKeys(IDLE);
        return effect == null ? null : effect - start;
    }

    async function pipeline(n = 15, opts = {}) {
        const out = [];
        for (let i = 0; i < n; i++) {
            const r = await pipelineOnce({ ...opts, action: i % 2 ? LEFT : RIGHT });
            if (r != null) out.push(r);
            await waitFrames(10);
            if (!alive()) { console.warn("died mid-test; stopping early"); break; }
        }
        const st = stats(out);
        console.log("raw pipeline lagFrames:", out.join(", "));
        console.table([st]);
        if (st) {
            console.log(
                `\n=> FULL pipeline (model + send + game): median ${st.median} frames ` +
                `(${(st.median * 1000 / TPS).toFixed(0)} ms).\n` +
                `   THIS is what the deployed agent actually experiences, so this is\n` +
                `   what SELF_LAG should be set from:\n` +
                `     SELF_LAG_MIN = ${st.min}\n` +
                `     SELF_LAG_MAX = ${st.max}\n` +
                `   Run selfRepeat() too and subtract: the difference is the model's\n` +
                `   share, i.e. how much you'd save by making inference faster.`);
        }
        return st;
    }

    function stop() {
        sendKeys(IDLE);
        if (echoHooked) { top.RECIEVEFUNCTION = prevRecv; echoHooked = false; }
        console.log("lagtest: keys released, hook removed.");
    }

    // ===== THE MAIN TOOL ======================================================
    // Evaluate the model on the LIVE observation, send its action, and report
    // the frame counter at entry and at exit. The gap is exactly what the model
    // costs per decision — measured directly, not inferred from watching the
    // ball move.
    //
    //   const [t0, t1] = await top.lagtest.send();
    //
    // t1 - t0 = frames burned by obs build + inference + presskeys.
    // Then note which frame the ball actually moves on and subtract t0 to get
    // the TOTAL self-lag the deployed agent really experiences.
    //
    // Pass an index to skip the model entirely: await top.lagtest.send(12).
    async function send(actionIndex) {
        const t0 = frame();
        let action = actionIndex, obsUsed = "explicit";
        if (action == null) {
            if (!policy) { console.warn("call await top.lagtest.loadModel() first"); return null; }
            hookKeys();
            const opp = opponentId();
            const nowTick = frame();
            let obs;
            if (opp != null && alive()) { obs = buildObs(top.myid, opp, nowTick); obsUsed = "live"; }
            else { obs = new Array(stateDim).fill(0); obsUsed = "zeros (no live opponent)"; }
            if (obs.length !== stateDim) {
                console.warn(`obs is ${obs.length} dims but the model wants ${stateDim}`);
            }
            action = await predict(obs);
        }
        sendKeys(action);
        const k = indexToKeys(action);
        lastSent = {
            up: k.up ? 1 : 0, down: k.down ? 1 : 0, left: k.left ? 1 : 0,
            right: k.right ? 1 : 0, heavy: k.heavy ? 1 : 0,
        };
        const t1 = frame();
        console.log(`send: frames [${t0}, ${t1}]  model cost ${t1 - t0} frame(s) ` +
                    `(${((t1 - t0) * 1000 / TPS).toFixed(0)} ms)  action ${action}  obs ${obsUsed}`);
        return [t0, t1];
    }

    top.lagtest.trace = trace;
    top.lagtest.send = send;          // async: evaluates the model, returns [t0, t1]
    top.lagtest.sendKeys = sendKeys;  // sync: raw keypress, no model
    top.lagtest.self = selfOnce;
    top.lagtest.selfRepeat = selfRepeat;
    top.lagtest.echo = echo;
    top.lagtest.loadModel = loadModel;
    top.lagtest.infer = infer;
    top.lagtest.pipeline = pipeline;
    top.lagtest.stop = stop;
    top.lagtest.frame = frame;

    console.log(
        "lagtest ready" + (frame() == null ? " — WARNING: top.getCurrentFrame() missing, nothing will work" : "") +
        ".\n" +
        "  await top.lagtest.loadModel()      -> pick models/bonk2-ppo-latest.json\n" +
        "  await top.lagtest.send()           -> model picks+sends; returns [t0, t1]\n" +
        "  await top.lagtest.trace(12)        -> RAW per-frame table of the effect\n" +
        "  await top.lagtest.selfRepeat(15)   -> game input lag alone\n" +
        "  await top.lagtest.loadModel()      -> pick models/bonk2-ppo-latest.json\n" +
        "  await top.lagtest.infer(50)        -> pure inference cost (ms + frames)\n" +
        "  await top.lagtest.pipeline(15)     -> model+send+game = the REAL SELF_LAG\n" +
        "  await top.lagtest.echo(15)         -> round-trip, estimates VIEW_LAG\n" +
        "  top.lagtest.stop()\n" +
        "Stand still in a live round with space to move, and stop play3.mjs first.\n" +
        "Load the EXPORTED slim model, not a raw runs/ checkpoint (~900 MB won't parse).");
})();
