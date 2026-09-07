// play7.mjs — paste-into-the-tab AI player for ppo7 checkpoints
// (python -m ppo7.train). Rebuilds the two-head (action + duration) TANH actor
// in plain JS, reads live bonk.io state, and drives your player with FiGAR holds.
//
// REQUIREMENTS (same as play3.mjs): the bonk.io instrumentation must already be
// injected (top.playerids, top.myid, top.scale, top.presskeys, top.MAKE_KEYS,
// top.GET_KEYS, top.RECIEVEFUNCTION, ideally top.getCurrentFrame). Paste AFTER
// it, pick the checkpoint in the dialog. Stop with top.bonkai.playStop().
//
// Differences from play3.mjs (bonk2/ppo6):
//   - Network is a TANH MLP *with biases*, NO LayerNorm (ppo7/networks.py).
//   - Actor has TWO heads: 18 action logits + 7 duration logits.
//   - The chosen action is HELD for DURATIONS[d] decision-cycles (FiGAR); the
//     net is only queried on a "free" cycle (when the hold expires).
//   - The net input is 34 = 33-dim obs + 1 appended "own remaining hold". The
//     actor only ever acts on a free cycle, so that feature is always 0 here.
//
// OBSERVATION — 33 dims, must match python/ppo7/env.py _frame() (NO time feat):
//   [ 0-11] self : x,y,vx,vy, heavyValue(masked), up,down,left,right,heavy,
//                  lastHeavySeen, ticksSinceSeen/200 (cap 1)
//   [12-23] opp  : same 12 (opponent's applied keys)
//   [24-27] rel  : dx,dy,dvx,dvy (normalized)
//   [28-32] pend : the keys WE LAST DECIDED
// Action index = lr*6 + ud*2 + heavy. Durations = [1,2,4,8,16,32,64] cycles.

(function () {
    "use strict";
    const isBrowser = (typeof top !== "undefined" && typeof document !== "undefined");
    if (isBrowser) top.bonkai = top.bonkai || {};

    // ===== Coordinate conversion (calibrated in play.mjs) ====================
    // "map by MaeIstrom": pixelsPerMeter = 12 (death=10, burger=15). ORIGIN is
    // the engine's fixed half-canvas (MAP_HALF_W/H px), same across all maps.
    const PPM = 10;
    const ORIGIN_X = 365, ORIGIN_Y = 250;
    const VEL_TIME = 1000;
    if (isBrowser) top.VEL_TIME = VEL_TIME;

    // ===== ppo7 constants (python/ppo7/config.py — keep in sync!) ============
    const POS_SCALE = 1 / 30;
    const VEL_SCALE = 1 / 30;
    const ACTION_REPEAT = 2;
    const HEAVY_SEEN_TICKS_NORM = 200;
    const RAW_STATE_DIM = 33;          // base obs before Fourier expansion
    // Fourier positional features (see python/ppo7/config.py FOURIER_*):
    // each pos/vel dim v -> [sin(pi 2^k v), cos(pi 2^k v)] for k in 0..L-1,
    // appended after the raw block. MUST match env.py _frame() exactly.
    const FOURIER_K_START = 4;   // dropped low octaves k=0..3 (redundant with raw)
    const FOURIER_L = 4;         // kept octaves k=4..7 (freq 16/32/64/128)
    const FOURIER_FEATS = 2 * FOURIER_L;                 // 8
    const FOURIER_BASE_DIMS = [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27];
    const STATE_DIM = RAW_STATE_DIM + FOURIER_BASE_DIMS.length * FOURIER_FEATS; // 225
    const AGENT_STATE_DIM = STATE_DIM + 1;               // net input = obs + hold feature = 226
    const NUM_ACTIONS = 18;
    const DURATIONS = [1, 2, 4, 8, 16, 32, 64];
    const TPS = 30;
    const DECIDE_MS = (1000 / TPS) * ACTION_REPEAT;
    const GREEDY = true;               // argmax both heads (false = sample)

    const HEAVY_CHILD = 2;
    const HEAVY_INVERT = false;

    const posX = (px) => (px / (isBrowser ? top.scale : 1) - ORIGIN_X) / PPM;
    const posY = (py) => (py / (isBrowser ? top.scale : 1) - ORIGIN_Y) / PPM;
    const velScale = () => VEL_TIME / ((isBrowser ? top.scale : 1) * PPM);

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

    // ===== Plain-JS inference (no TensorFlow / no CDN) =======================
    // Kernel in tfjs record layout: flat [in, out], row-major. y = xW + b.
    function matvec(kernel, inDim, outDim, x, bias) {
        const y = new Float32Array(outDim);
        if (bias) y.set(bias);
        for (let i = 0; i < inDim; i++) {
            const xi = x[i];
            if (xi === 0) continue;
            const row = i * outDim;
            for (let j = 0; j < outDim; j++) y[j] += xi * kernel[row + j];
        }
        return y;
    }

    // ppo7 records: [k0,b0, k1,b1, ..., k_out,b_out]; each Linear = kernel[in,out]
    // + bias[out]; tanh between hidden layers; the output layer has no activation.
    function buildMLP(records) {
        const layers = [];
        for (let i = 0; i < records.length - 2; i += 2) {
            layers.push({
                k: Float32Array.from(records[i].data),
                inDim: records[i].shape[0], outDim: records[i].shape[1],
                b: Float32Array.from(records[i + 1].data),
            });
        }
        const oi = records.length - 2;
        return {
            layers,
            outK: Float32Array.from(records[oi].data),
            outIn: records[oi].shape[0], outOut: records[oi].shape[1],
            outB: Float32Array.from(records[oi + 1].data),
            inDim: layers.length ? layers[0].inDim : records[oi].shape[0],
            outDim: records[oi].shape[1],
        };
    }

    function forward(net, obs) {
        let x = Float32Array.from(obs);
        for (const L of net.layers) {
            const y = matvec(L.k, L.inDim, L.outDim, x, L.b);
            for (let i = 0; i < y.length; i++) y[i] = Math.tanh(y[i]);
            x = y;
        }
        return matvec(net.outK, net.outIn, net.outOut, x, net.outB);
    }

    function argmax(a) {
        let b = 0;
        for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
        return b;
    }
    function sampleSoftmax(a) {
        let m = -Infinity;
        for (const v of a) if (v > m) m = v;
        let sum = 0;
        const p = new Float64Array(a.length);
        for (let i = 0; i < a.length; i++) { p[i] = Math.exp(a[i] - m); sum += p[i]; }
        let r = Math.random() * sum;
        for (let i = 0; i < a.length; i++) { r -= p[i]; if (r <= 0) return i; }
        return a.length - 1;
    }

    // Stateful (action, duration) decider with the FiGAR hold. Only queries the
    // net on a FREE cycle (hold expired); repeats the held action otherwise.
    function makeDecider(net, greedy) {
        let holdRemaining = 0, heldAction = 0, lastDur = 0;
        return {
            decide(obs33) {
                if (holdRemaining > 0) {
                    holdRemaining--;
                    return { action: heldAction, held: true, dur: lastDur };
                }
                const obs = Array.from(obs33);
                obs.push(0.0);                       // free cycle -> hold feature 0
                const logits = forward(net, obs);
                const aL = logits.subarray(0, NUM_ACTIONS);
                const dL = logits.subarray(NUM_ACTIONS);
                const a = greedy ? argmax(aL) : sampleSoftmax(aL);
                const d = greedy ? argmax(dL) : sampleSoftmax(dL);
                heldAction = a;
                lastDur = DURATIONS[d];
                holdRemaining = lastDur - 1;         // this cycle counts as 1
                return { action: a, held: false, dur: lastDur };
            },
            reset() { holdRemaining = 0; heldAction = 0; lastDur = 0; },
        };
    }

    // ---- node test hook: export the pure functions, do NOT run the browser loop
    if (!isBrowser) {
        globalThis.__play7 = {
            buildMLP, forward, argmax, sampleSoftmax, makeDecider, indexToKeys,
            DURATIONS, NUM_ACTIONS, STATE_DIM, AGENT_STATE_DIM,
        };
        return;
    }

    // ===== Local state (browser) ============================================
    const keyMap = new Map();
    const heavyTrack = new Map();
    let policy = null, decider = null, running = false;
    let lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
    let lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 };
    let roundStartFrame = null, roundStartMs = 0;

    function appliedKeys(id) { return top.GET_KEYS(keyMap.get(id) ?? 0); }
    function heavyAlpha(id) {
        let a = 0;
        try { a = top.playerids[id].playerData.children[HEAVY_CHILD].alpha; } catch (_) { a = 0; }
        return HEAVY_INVERT ? 1 - a : a;
    }
    function elapsedTicks() {
        if (typeof top.getCurrentFrame === "function" && roundStartFrame != null)
            return Math.max(0, top.getCurrentFrame() - roundStartFrame);
        return (performance.now() - roundStartMs) / (1000 / TPS);
    }
    function heavyFeatures(id, heavyHeld) {
        const now = elapsedTicks();
        let t = heavyTrack.get(id);
        if (!t) { t = { seen: 1, seenAtTick: 0 }; heavyTrack.set(id, t); }
        if (heavyHeld) { t.seen = heavyAlpha(id); t.seenAtTick = now; }
        const ticksSince = Math.max(0, now - t.seenAtTick);
        return [t.seen, Math.min(1, ticksSince / HEAVY_SEEN_TICKS_NORM)];
    }
    function block(id) {
        const d = top.playerids[id].playerData2;
        const k = appliedKeys(id);
        const vs = velScale();
        const heavyValue = k.heavy ? heavyAlpha(id) : 0;
        const [seen, sinceNorm] = heavyFeatures(id, !!k.heavy);
        return [
            posX(d.px) * POS_SCALE, posY(d.py) * POS_SCALE,
            d.xvel * vs * VEL_SCALE, d.yvel * vs * VEL_SCALE,
            heavyValue,
            k.up ? 1 : 0, k.down ? 1 : 0, k.left ? 1 : 0, k.right ? 1 : 0, k.heavy ? 1 : 0,
            seen, sinceNorm,
        ];
    }
    // Append NeRF-style Fourier features for the pos/vel dims (raw kept).
    // Layout per base dim: sin_0,cos_0,sin_1,cos_1,... — must match env.py.
    function fourierExpand(obs33) {
        const out = obs33.slice();               // raw 33 stay in place
        for (const d of FOURIER_BASE_DIMS) {
            const v = obs33[d];
            for (let k = 0; k < FOURIER_L; k++) {
                const a = Math.PI * Math.pow(2, FOURIER_K_START + k) * v;
                out.push(Math.sin(a), Math.cos(a));
            }
        }
        return out;                              // length 225
    }
    // 33-dim raw obs (NO time feature) + Fourier expansion -> 225 dims.
    function buildObs(meId, oppId) {
        const md = top.playerids[meId].playerData2;
        const od = top.playerids[oppId].playerData2;
        const vs = velScale();
        const ds = 1 / (top.scale * PPM);
        const raw = block(meId).concat(block(oppId), [
            (od.px - md.px) * ds * POS_SCALE,
            (od.py - md.py) * ds * POS_SCALE,
            (od.xvel - md.xvel) * vs * VEL_SCALE,
            (od.yvel - md.yvel) * vs * VEL_SCALE,
            lastSent.up, lastSent.down, lastSent.left, lastSent.right, lastSent.heavy,
        ]);
        return fourierExpand(raw);
    }

    function playMove(actionIndex) {
        const keys = indexToKeys(actionIndex);
        top.presskeys(lastMove, keys);
        keyMap.set(top.myid, top.MAKE_KEYS(keys));
        lastMove = { ...keys };
        lastSent = {
            up: keys.up ? 1 : 0, down: keys.down ? 1 : 0,
            left: keys.left ? 1 : 0, right: keys.right ? 1 : 0, heavy: keys.heavy ? 1 : 0,
        };
    }

    top.RECIEVEFUNCTION = function (args) {
        if (args.startsWith("42[7,")) {
            const data = JSON.parse(args.slice(2));
            keyMap.set(data[1], data[2].i);
        }
        if (args.startsWith("42[15,")) {
            top.presskeys(lastMove, { up: false, down: false, left: false, right: false, heavy: false, special: false });
            for (let i in top.playerids) top.playerids[i].playerData2.alive = false;
        }
        return args;
    };

    function isAlive(id) {
        const p = top.playerids && top.playerids[id];
        return !!(p && p.playerData2 && p.playerData2.alive);
    }
    function getIds() {
        const me = top.myid;
        if (!isAlive(me)) return null;
        for (const pid in top.playerids)
            if (pid != me && isAlive(pid)) return [me, parseInt(pid)];
        return null;
    }

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

    async function main() {
        const save = await loadFile();
        const agent = save.agent || save;
        if (!agent.actor) throw new Error(`no actor weights (agent keys: ${Object.keys(agent)})`);
        policy = buildMLP(agent.actor);
        decider = makeDecider(policy, GREEDY);
        top.bonkai.policy = policy;
        if (policy.inDim !== AGENT_STATE_DIM)
            console.warn(`expected net input ${AGENT_STATE_DIM}, save says ${policy.inDim} — wrong model? (ppo6 uses play3.mjs)`);
        if (policy.outDim !== NUM_ACTIONS + DURATIONS.length)
            console.warn(`expected ${NUM_ACTIONS + DURATIONS.length} outputs (18 action + 7 duration), got ${policy.outDim}`);
        console.log(`ppo7 AI loaded ${policy.inDim}->${policy.outDim} (2 heads)` +
            (save.progress ? `, ELO ${Math.round(save.progress.currentRating)}` : "") +
            `. Deciding every ${ACTION_REPEAT} frames (${GREEDY ? "greedy" : "sampled"}), FiGAR holds. Stop: top.bonkai.playStop().`);

        let inRound = false;
        running = true;
        while (running) {
            const ids = getIds();
            if (ids) {
                if (!inRound) {
                    inRound = true;
                    roundStartMs = performance.now();
                    roundStartFrame = typeof top.getCurrentFrame === "function" ? top.getCurrentFrame() : null;
                    heavyTrack.clear();
                    decider.reset();               // new round -> free decision
                }
                const [me, opp] = ids;
                let action = 0;
                try { action = decider.decide(buildObs(me, opp)).action; }
                catch (err) { console.error("inference error:", err); }
                playMove(action);
            } else if (inRound) {
                inRound = false;
                top.presskeys(lastMove, { up: false, down: false, left: false, right: false, heavy: false, special: false });
                lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
                lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 };
                heavyTrack.clear();
                decider.reset();
            }
            await sleep(DECIDE_MS);
        }
        console.log("AI stopped.");
    }

    top.bonkai.playStop = () => { running = false; };
    top.bonkai.playRestart = () => { if (!running) main().catch((e) => console.error(e)); };
    main().catch((e) => console.error("play7.mjs failed:", e));
})();
