// play3.mjs — paste-into-the-tab AI player for bonk2 (v2) checkpoints
// (python -m bonk2.train). Prompts for the checkpoint JSON, rebuilds the actor
// with TensorFlow.js, reads live bonk.io state, and drives your player.
//
// REQUIREMENTS (same as play2.mjs)
//   1. The bonk.io instrumentation from Bonk1v1Ai must already be injected:
//      top.playerids, top.myid, top.scale, top.presskeys, top.MAKE_KEYS,
//      top.GET_KEYS, top.RECIEVEFUNCTION (and ideally top.getCurrentFrame).
//   2. Paste AFTER the instrumentation; pick the checkpoint in the file dialog.
//      Stop with top.bonkai.playStop().
//
// No network access needed: inference is plain JS, no TensorFlow, no CDN.
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
    const PPM = 10;                    // map1.json physics.ppm
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
    let policy = null;
    let hiddenState = null;   // GRU state; null for feedforward
    let running = false;
    let lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
    let lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 }; // pending block
    let roundStartFrame = null;        // for the draw clock + heavy tracker
    let roundStartMs = 0;

    // ===== Inference: plain JS, no TensorFlow =================================
    // The actor is ~79k MACs. Pulling 1 MB of tfjs from a CDN to run that cost
    // us a WebGL backend fighting the game's renderer, an async readback worth
    // ~2 frames of input lag, and (for the GRU) Keras's gate order silently
    // disagreeing with PyTorch's. This is the same code as
    // python/bonk3/tools/net.mjs, which is parity-tested against PyTorch to
    // ~1e-8; keep the two in sync (tools/test_net_parity.mjs checks for drift).
    // Bonus: the script is now fully self-contained — no network dependency.

    // ── primitives ───────────────────────────────────────────────────────────────

    // Kernel in tfjs record layout: flat [in, out], row-major. y = xW.
    function matvecTfjs(kernel, inDim, outDim, x, bias) {
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

    // Torch Linear layout: nested [out][in]. y = Wx + b.
    function matvecTorch(W, x, bias) {
        const outDim = W.length;
        const y = new Float32Array(outDim);
        for (let j = 0; j < outDim; j++) {
            const row = W[j];
            let s = bias ? bias[j] : 0;
            for (let i = 0; i < row.length; i++) s += row[i] * x[i];
            y[j] = s;
        }
        return y;
    }

    // Matches torch.nn.LayerNorm: biased variance (divide by N, not N-1).
    function layerNorm(x, gamma, beta, eps = 1e-3) {
        const n = x.length;
        let mean = 0;
        for (let i = 0; i < n; i++) mean += x[i];
        mean /= n;
        let varc = 0;
        for (let i = 0; i < n; i++) { const d = x[i] - mean; varc += d * d; }
        varc /= n;
        const inv = 1 / Math.sqrt(varc + eps);
        const y = new Float32Array(n);
        for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
        return y;
    }

    function relu(x) {
        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) y[i] = x[i] > 0 ? x[i] : 0;
        return y;
    }

    const sigmoid = (v) => 1 / (1 + Math.exp(-v));

    // ── feedforward actor (tfjs record format) ───────────────────────────────────
    // Records: k0, ln0_g, ln0_b, k1, ln1_g, ln1_b, ..., k_out, b_out.
    // Layer block = [Linear(no bias) -> LayerNorm(eps 1e-3) -> ReLU], then a final
    // Linear WITH bias. Same structure as bonk/networks.py MLP.
    function buildMLP(records) {
        const layers = [];
        let i = 0;
        while (i + 3 < records.length) {           // a trailing kernel+bias remain
            layers.push({
                k: Float32Array.from(records[i].data),
                inDim: records[i].shape[0], outDim: records[i].shape[1],
                g: Float32Array.from(records[i + 1].data),
                b: Float32Array.from(records[i + 2].data),
            });
            i += 3;
        }
        const outK = records[i], outB = records[i + 1];
        return {
            layers,
            outK: Float32Array.from(outK.data),
            outIn: outK.shape[0], outOut: outK.shape[1],
            outB: Float32Array.from(outB.data),
            stateDim: layers.length ? layers[0].inDim : outK.shape[0],
            actionDim: outK.shape[1],
        };
    }

    function mlpForward(net, obs) {
        let x = Float32Array.from(obs);
        for (const L of net.layers) {
            x = relu(layerNorm(matvecTfjs(L.k, L.inDim, L.outDim, x, null), L.g, L.b));
        }
        return matvecTfjs(net.outK, net.outIn, net.outOut, x, net.outB);
    }

    // ── recurrent actor (RecurrentAC.to_records) ─────────────────────────────────
    function buildGRU(obj) {
        const t = obj.tensors;
        const enc = [];
        // enc is Sequential(Linear(no bias), LayerNorm, ReLU) repeated.
        for (let li = 0; ; li += 3) {
            if (!(`enc.${li}.weight` in t)) break;
            enc.push({
                W: t[`enc.${li}.weight`],
                g: t[`enc.${li + 1}.weight`],
                b: t[`enc.${li + 1}.bias`],
            });
        }
        return {
            enc,
            Wih: t["cell.weight_ih"], Whh: t["cell.weight_hh"],
            bih: t["cell.bias_ih"], bhh: t["cell.bias_hh"],
            piW: t["pi.weight"], piB: t["pi.bias"],
            hidden: obj.hidden, stateDim: obj.stateDim, actionDim: obj.numActions,
        };
    }

    // EXACT torch.nn.GRUCell semantics. Gate order in weight_ih/weight_hh is
    // [r, z, n]; crucially the reset gate multiplies (W_hn h + b_hn) INCLUDING the
    // hidden bias. Keras orders gates (z, r, h) and applies the reset differently —
    // that mismatch is the classic silent-corruption bug this avoids.
    function gruStep(net, obs, h) {
        let x = Float32Array.from(obs);
        for (const L of net.enc) x = relu(layerNorm(matvecTorch(L.W, x, null), L.g, L.b));

        const H = net.hidden;
        const gi = matvecTorch(net.Wih, x, net.bih);   // [3H]
        const gh = matvecTorch(net.Whh, h, net.bhh);   // [3H]
        const hn = new Float32Array(H);
        for (let j = 0; j < H; j++) {
            const r = sigmoid(gi[j] + gh[j]);
            const z = sigmoid(gi[H + j] + gh[H + j]);
            const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
            hn[j] = (1 - z) * n + z * h[j];
        }
        return { logits: matvecTorch(net.piW, hn, net.piB), h: hn };
    }

    // ── unified entry ────────────────────────────────────────────────────────────
    function buildPolicy(save) {
        const agent = save.agent || save;
        if (agent.recurrent) {
            const net = buildGRU(agent.recurrent);
            return { kind: "gru", net, hidden: net.hidden,
                     stateDim: net.stateDim, actionDim: net.actionDim };
        }
        const net = buildMLP(agent.actor);
        return { kind: "mlp", net, hidden: 0,
                 stateDim: net.stateDim, actionDim: net.actionDim };
    }

    function policyStep(p, obs, h) {
        if (p.kind === "gru") return gruStep(p.net, obs, h);
        return { logits: mlpForward(p.net, obs), h };
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

    // One action index from the current observation, advancing GRU state.
    function decide(obs) {
        const r = policyStep(policy, obs, hiddenState);
        hiddenState = r.h;
        return GREEDY ? argmax(r.logits) : sampleSoftmax(r.logits);
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
        if (!agent.actor && !agent.recurrent) {
            throw new Error(`no actor/recurrent weights in save (agent keys: ${Object.keys(agent)})`);
        }
        policy = buildPolicy(save);
        hiddenState = policy.kind === "gru" ? new Float32Array(policy.hidden) : null;
        top.bonkai.policy = policy;
        if (policy.stateDim !== STATE_DIM) {
            console.warn(`expected obs dim ${STATE_DIM}, save says ${policy.stateDim}` +
                         " — v1 model? use play2.mjs for those");
        }
        console.log(
            `AI loaded [${policy.kind}] ${policy.stateDim}->${policy.actionDim}` +
            (policy.kind === "gru" ? ` (hidden ${policy.hidden})` : "") +
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
                    actionIndex = decide(buildObs(me, opp));
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
