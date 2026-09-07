// play9.mjs — paste-into-the-tab AI player for ppo9 checkpoints
// (python -m ppo9.train, ideally shrunk with `python -m ppo9.compress`).
// Rebuilds the two-head (action + duration) minGRU actor in plain JS, reads
// live bonk.io state, and drives your player with FiGAR holds.
//
// REQUIREMENTS (same as play3.mjs): the bonk.io instrumentation must already be
// injected (top.playerids, top.myid, top.scale, top.presskeys, top.MAKE_KEYS,
// top.GET_KEYS, top.RECIEVEFUNCTION, ideally top.getCurrentFrame). Paste AFTER
// it, pick the checkpoint in the dialog. Stop with top.bonkai.playStop().
//
// Differences from play7.mjs (ppo7) — ONLY the network and its input encoding
// changed; the environment collection below is untouched:
//   - Network is a minGRU RECURRENT encoder feeding a tanh MLP
//     (python/ppo9/mingru.py):  z, h~ = split(W_gru x + b);  z = sigmoid(z);
//     h_t = (1-z) h_{t-1} + z h~;  out = W_out tanh_mlp([x || h_t]).
//   - THE MEMORY STEPS EVERY CYCLE, the hold only gates which action is
//     APPLIED. Querying the net only on free cycles (what play7 did, correctly,
//     for a feedforward net) would starve the recurrence and is not a small
//     effect: measured on step8.9B, a memory-carrying seat beats an otherwise
//     identical memory-zeroed seat 95.5% of decisive games.
//   - The hold feature is therefore hold/max(DURATIONS), NOT always 0 — on a
//     held cycle it is nonzero, exactly as in ppo9/policy.py ActorPolicy.act.
//   - Fourier features use L=4 octaves from k=4 (ppo7 used L=8 from k=0), so
//     the observation is 33 + 12*8 = 129 and the net input 130.
//
// OBSERVATION — 33 dims, must match python/bonkenv/env.py _raw_frame():
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

    // ===== One instance at a time ===========================================
    // Pasting the script again must REPLACE the running one, not race it: two
    // loops both calling top.presskeys fight over the same player and the
    // inputs interleave into nonsense.
    //
    // A generation token rather than only calling the old playStop(): that
    // closure belongs to the previous paste and may be from an older version
    // of this file, or already broken. Every loop re-checks the token it was
    // born with and exits the moment a newer paste claims it, so the old
    // instance retires itself even if nothing could reach into it.
    const MY_ID = isBrowser ? (top.bonkai.instanceId || 0) + 1 : 1;
    if (isBrowser) {
        const prev = top.bonkai.instanceId;
        top.bonkai.instanceId = MY_ID;
        if (prev) {
            console.log(`[play9] replacing instance #${prev} -> #${MY_ID}`);
            try { top.bonkai.playStop?.(); } catch (_) { /* old closure, ignore */ }
            // Whatever the old loop was holding down, let go of it — otherwise
            // a direction stays pressed until the new loop happens to set it.
            try {
                top.presskeys(
                    { left: true, right: true, up: true, down: true, heavy: true, special: true },
                    { left: false, right: false, up: false, down: false, heavy: false, special: false });
            } catch (_) { /* instrumentation not ready yet */ }
        }
    }
    const isCurrent = () => !isBrowser || top.bonkai.instanceId === MY_ID;

    // ===== Coordinate conversion (calibrated in play.mjs) ====================
    const PPM = 10;
    const ORIGIN_X = 365, ORIGIN_Y = 250;
    const VEL_TIME = 1000;
    if (isBrowser) top.VEL_TIME = VEL_TIME;

    // ===== ppo9 constants (python/ppo9/config.py — keep in sync!) ============
    const POS_SCALE = 1 / 30;
    const VEL_SCALE = 1 / 30;
    const ACTION_REPEAT = 2;
    const HEAVY_SEEN_TICKS_NORM = 200;
    const RAW_STATE_DIM = 33;          // base obs before Fourier expansion
    // Fourier positional features (bonkenv/config.py ObsConfig.fourier_*):
    // each pos/vel dim v -> [sin(f v), cos(f v)] for f = pi*2^k,
    // k = K_START .. K_START+L-1, appended after the raw block.
    // MUST match bonkenv/obs.py fourier_expand exactly.
    const FOURIER_L = 4;
    const FOURIER_K_START = 4;                           // ppo7 used L=8 from k=0
    const FOURIER_FEATS = 2 * FOURIER_L;                 // 8
    const FOURIER_BASE_DIMS = [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27];
    const STATE_DIM = RAW_STATE_DIM + FOURIER_BASE_DIMS.length * FOURIER_FEATS; // 129
    const AGENT_STATE_DIM = STATE_DIM + 1;               // net input = obs + hold = 130
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

    // ppo9 minGRU records (python/ppo9/mingru.py to_records):
    //   { type:"mingru", meta:{in_dim,hidden,out_dim,gru_hidden},
    //     gru:[kernel,bias],                       Linear(in_dim -> 2*H)
    //     mlp:[k0,b0, k1,b1, ..., kOut,bOut] }     tanh between hidden layers,
    // where the MLP's first layer takes [x || h] so its input is in_dim + H,
    // and the LAST pair is the output layer (no activation).
    function buildNet(records) {
        const dense = (r, i) => ({
            k: Float32Array.from(r[i].data),
            inDim: r[i].shape[0], outDim: r[i].shape[1],
            b: Float32Array.from(r[i + 1].data),
        });
        const m = records.mlp;
        const layers = [];
        for (let i = 0; i < m.length - 2; i += 2) layers.push(dense(m, i));
        const oi = m.length - 2;
        return {
            gru: dense(records.gru, 0),
            layers,
            out: dense(m, oi),
            H: records.meta.gru_hidden,
            inDim: records.meta.in_dim,
            outDim: records.meta.out_dim,
        };
    }

    // One recurrent step. `h` is mutated in place and is the ONLY state that
    // must survive between cycles — see the note about stepping every cycle.
    function stepNet(net, x, h) {
        const H = net.H;
        const zt = matvec(net.gru.k, net.gru.inDim, net.gru.outDim, x, net.gru.b);
        const h2 = new Float32Array(H);
        for (let i = 0; i < H; i++) {
            const z = 1 / (1 + Math.exp(-zt[i]));        // sigmoid gate
            h2[i] = (1 - z) * h[i] + z * zt[H + i];      // h~ is NOT squashed
        }
        // The head reads [x || h_t] — the POST-update hidden, so the output
        // reflects the current observation (mingru.py step()).
        const cat = new Float32Array(x.length + H);
        cat.set(x, 0); cat.set(h2, x.length);
        let a = cat;
        for (const L of net.layers) {
            const y = matvec(L.k, L.inDim, L.outDim, a, L.b);
            for (let i = 0; i < y.length; i++) y[i] = Math.tanh(y[i]);
            a = y;
        }
        h.set(h2);
        return matvec(net.out.k, net.out.inDim, net.out.outDim, a, net.out.b);
    }

    function softmax(a) {
        let m = -Infinity;
        for (const v of a) if (v > m) m = v;
        let sum = 0;
        const p = new Float64Array(a.length);
        for (let i = 0; i < a.length; i++) { p[i] = Math.exp(a[i] - m); sum += p[i]; }
        for (let i = 0; i < p.length; i++) p[i] /= sum;
        return p;
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


    // ===== Chat ==============================================================
    // Driven by the CRITIC, not the actor: [win, draw, loss] from OUR seat's
    // point of view, once per decision cycle (~15 Hz). Every rule is a row in
    // DEFAULT_CHAT.rules so adding a situation is data, not code.
    //
    // Three independent anti-spam layers, because any one alone is not enough:
    //   globalCooldownMs  nothing may follow anything else too closely
    //   rule.cooldownMs   the SAME line cannot repeat (stalling: 60s, as asked)
    //   perRoundMax       a hard ceiling per round regardless of cooldowns
    // plus `holdCycles` per rule, so a level condition must PERSIST rather than
    // fire on one noisy frame, and `warmupCycles`, because the value estimate
    // right after a spawn is meaningless.
    //
    // Swing rules read a ring of the last `windowMs` of win-probability, so
    // "rapidly increased over the last second" is measured, not guessed.
    const DEFAULT_CHAT = {
        globalCooldownMs: 9000,
        perRoundMax: 3,
        warmupCycles: 25,          // ~1.7s of settling before anything may fire
        windowMs: 1000,            // the swing window
        // MEASURED, not guessed. On the live checkpoint the critic sits at
        // pWin ~0.51 almost always (p05 0.486, p95 0.549), so "a big swing" is
        // rarer than it feels — but at 15 Hz even a 0.5%-of-windows event fires
        // every ~13s, which is why 0.35 produced a running commentary of
        // "oh you messed up" while nobody was actually losing. At 0.50 that
        // becomes ~1 per 42s AND 100% of such swings land somewhere decisive;
        // 0.55 with the band below is stricter still.
        swingUp: 0.55,             // win-prob RISE that counts as "they blundered"
        swingDown: 0.55,
        // ...and the swing must END somewhere that a blunder claim makes sense.
        // Without this, 19% of qualifying swings finished inside 0.35-0.65 —
        // i.e. the exact "no side is losing" case that reads as nonsense.
        decisiveBand: 0.15,        // |pWin - 0.5| required after the swing
        drawSwing: 0.30,           // fast change in draw prob = stall starting/breaking
        drawHigh: 0.50,            // pDraw above this reads as a stall
        drawLow: 0.10,             // below this nobody is stalling
        sureWin: 0.90,
        sureLoss: 0.12,
        crushing: 0.97,            // beyond "winning" — it is not close
        hopeless: 0.03,
        earlyCycles: 120,          // ~8s: still the opening
        longCycles: 900,           // ~60s: this round is dragging
        seesawSpan: 0.70,          // roundMax-roundMin that counts as a wild game
        streakLen: 3,              // rounds in a row before it gets mentioned
        // null -> `top.chat(msg)`. Set this to route lines somewhere else.
        send: null,
        // ORDER MATTERS: the tick loop fires at most one line per cycle and
        // stops at the first match, so the most specific/urgent rules sit
        // first and the ambient commentary last.
        rules: [
            // --- edge-triggered: a fast change in who is winning -------------
            { name: "theyBlundered", kind: "swingUp", holdCycles: 5,
              cooldownMs: 30000, lines: [
                "oh you messed up", "oops", "that's rough buddy", "yikes",
                "what was that", "i'll take it", "thanks for that" ] },
            { name: "iBlundered", kind: "swingDown", holdCycles: 5,
              cooldownMs: 30000, lines: [
                "oh shoot i lost", "nice one", "ugh", "okay that was good",
                "how did i fall for that", "my bad", "well played" ] },
            // --- the stall, starting / holding / breaking --------------------
            { name: "drawSpike", kind: "drawSpike", cooldownMs: 45000, lines: [
                "don't you dare stall", "here we go", "oh you're running now",
                "really?" ] },
            { name: "stalling", kind: "drawHigh", holdCycles: 45, cooldownMs: 60000,
              lines: [ "you are stalling", "you are actually stalling",
                       "stop stalling", "this is stalling and you know it",
                       "fight me" ] },
            { name: "stalemateBroken", kind: "stalemateBroken", cooldownMs: 45000,
              lines: [ "finally", "oh NOW you want to play", "there we go",
                       "was that so hard" ] },
            // --- extremes ----------------------------------------------------
            { name: "dominating", kind: "crushing", holdCycles: 60, cooldownMs: 60000,
              lines: [ "are you even trying", "this is not close",
                       "i can do this all day", "sorry" ] },
            { name: "hopeless", kind: "hopeless", holdCycles: 60, cooldownMs: 60000,
              lines: [ "yeah that's game", "you got me", "i give up",
                       "teach me that" ] },
            { name: "winning", kind: "sureWin", holdCycles: 30, cooldownMs: 45000,
              lines: [ "gg", "this one's over", "i've got this", "calling it now" ] },
            { name: "losing", kind: "sureLoss", holdCycles: 30, cooldownMs: 45000,
              lines: [ "i'm cooked", "gg wp", "this is rough", "not looking great" ] },
            // --- whole-round arcs --------------------------------------------
            { name: "comeback", kind: "comeback", cooldownMs: 60000, lines: [
                "and we're back", "never in doubt", "don't count me out",
                "told you" ] },
            { name: "clutch", kind: "clutch", cooldownMs: 60000, lines: [
                "still alive", "not done yet", "we move" ] },
            { name: "threwIt", kind: "threwIt", cooldownMs: 60000, lines: [
                "how did i lose that", "i had that", "i threw it",
                "that one hurts" ] },
            { name: "seesaw", kind: "seesaw", cooldownMs: 60000, lines: [
                "what a game", "this is wild", "back and forth" ] },
            { name: "earlyLead", kind: "earlyLead", cooldownMs: 60000, lines: [
                "good start for me", "off to a good start", "early lead" ] },
            // --- ambient ------------------------------------------------------
            { name: "evenFight", kind: "evenFight", holdCycles: 600, cooldownMs: 240000,
              lines: [ "dead even", "someone has to break first",
                       "we're the same", "even so far" ] },
            { name: "closeGame", kind: "close", holdCycles: 900, cooldownMs: 240000,
              lines: [ "this is close", "good game so far", "tight one" ] },
            { name: "longRound", kind: "longRound", holdCycles: 60, cooldownMs: 240000, lines: [
                "this is taking forever", "are we done yet",
                "someone end this" ] },
            // --- round boundaries ---------------------------------------------
            { name: "greeting", kind: "firstRound", cooldownMs: Infinity, lines: [
                "gl hf", "glhf", "good luck" ] },
            { name: "winStreak", kind: "winStreak", cooldownMs: 60000, lines: [
                "that's three", "again?", "same result", "you know how this ends" ] },
            { name: "lossStreak", kind: "lossStreak", cooldownMs: 60000, lines: [
                "okay you're just better", "how are you doing that",
                "teach me", "i need a break" ] },
            { name: "roundWin", kind: "wonRound", cooldownMs: 30000, lines: [
                "gg", "ez", "gg wp", "good round" ] },
            { name: "roundLoss", kind: "lostRound", cooldownMs: 30000, lines: [
                "gg wp", "good one", "nice", "well played" ] },
        ],
    };

    // The page provides `top.chat(string)`. `top.chatConfig.send` overrides it.
    function defaultSend(msg) {
        top.chat(msg);
        return true;
    }

    function makeChatter(getCfg) {
        let ring = [];             // [{t, pWin, pDraw}] over the last windowMs
        let winStreak = 0, lossStreak = 0;
        // -Infinity, not 0: with 0 the global-cooldown check suppresses the
        // very first message whenever the clock starts near zero.
        let lastSentAt = -Infinity, sentThisRound = 0, cycles = 0, rounds = 0;
        let roundMin = 1, roundMax = 0;
        const holds = new Map();   // rule -> consecutive cycles its condition held
        const lastFired = new Map();
        const history = [];

        const cond = (kind, cfg, pWin, pDraw, swing, dSwing) => {
            switch (kind) {
                // Both halves required: a big MOVE that also ENDS somewhere
                // one side is plainly ahead. A large swing that returns to even
                // is the critic twitching, not a blunder.
                case "swingUp":
                    return swing >= cfg.swingUp && pWin >= 0.5 + cfg.decisiveBand;
                case "swingDown":
                    return swing <= -cfg.swingDown && pWin <= 0.5 - cfg.decisiveBand;
                case "drawSpike": return dSwing >= cfg.drawSwing && pDraw >= cfg.drawHigh;
                case "drawHigh": return pDraw >= cfg.drawHigh;
                case "stalemateBroken":
                    return dSwing <= -cfg.drawSwing && pDraw <= cfg.drawLow;
                case "crushing": return pWin >= cfg.crushing;
                case "hopeless": return pWin <= cfg.hopeless;
                case "sureWin": return pWin >= cfg.sureWin;
                case "sureLoss": return pWin <= cfg.sureLoss;
                case "close": return pWin > 0.42 && pWin < 0.58;
                // Distinct from `close`: a genuinely even FIGHT, not a
                // stalemate that happens to sit near 50%.
                case "evenFight":
                    return pWin > 0.45 && pWin < 0.55 && pDraw <= cfg.drawLow;
                case "comeback": return roundMin < 0.25 && pWin > 0.62;
                case "clutch": return roundMin < 0.15 && pWin > 0.50;
                case "threwIt": return roundMax > 0.78 && pWin < 0.38;
                case "seesaw": return roundMax - roundMin >= cfg.seesawSpan;
                case "earlyLead": return cycles <= cfg.earlyCycles && pWin >= 0.80;
                case "longRound": return cycles >= cfg.longCycles;
                default: return false;          // round-boundary kinds only
            }
        };

        function emit(rule, now, cfg) {
            const line = rule.lines[Math.floor(Math.random() * rule.lines.length)];
            const fn = cfg.send || defaultSend;
            try { fn(line); } catch (e) { console.error("chat send failed:", e); }
            lastSentAt = now; sentThisRound++;
            lastFired.set(rule.name, now);
            history.push({ t: now, rule: rule.name, line });
            if (history.length > 50) history.shift();
        }

        function fire(kind, now, cfg) {
            if (!cfg.enabled) return;
            const rule = cfg.rules.find((r) => r.kind === kind);
            if (!rule) return;
            if (now - lastSentAt < cfg.globalCooldownMs) return;
            if (now - (lastFired.get(rule.name) ?? -Infinity) < rule.cooldownMs) return;
            emit(rule, now, cfg);
        }

        return {
            // Called once per decision with the critic's [win, draw, loss].
            tick(probs) {
                const cfg = getCfg();
                if (!probs) return;
                const now = (typeof performance !== "undefined") ? performance.now() : Date.now();
                const pWin = probs[0], pDraw = probs[1];
                ring.push({ t: now, pWin, pDraw });
                while (ring.length && now - ring[0].t > cfg.windowMs) ring.shift();
                cycles++;
                roundMin = Math.min(roundMin, pWin);
                roundMax = Math.max(roundMax, pWin);
                if (!cfg.enabled || cycles < cfg.warmupCycles) return;
                if (sentThisRound >= cfg.perRoundMax) return;

                const swing = pWin - ring[0].pWin;     // over the last windowMs
                const dSwing = pDraw - ring[0].pDraw;  // ...and for the draw share
                for (const rule of cfg.rules) {
                    if (["firstRound", "wonRound", "lostRound",
                         "winStreak", "lossStreak"].includes(rule.kind)) continue;
                    const ok = cond(rule.kind, cfg, pWin, pDraw, swing, dSwing);
                    const need = rule.holdCycles ?? 1;
                    const n = ok ? (holds.get(rule.name) ?? 0) + 1 : 0;
                    holds.set(rule.name, n);
                    if (n < need) continue;
                    if (now - lastSentAt < cfg.globalCooldownMs) continue;
                    if (now - (lastFired.get(rule.name) ?? -Infinity) < rule.cooldownMs) continue;
                    emit(rule, now, cfg);
                    holds.set(rule.name, 0);
                    // A swing is read off a ROLLING window, so the same
                    // excursion stays qualifying for another windowMs. Drop the
                    // window so the next line needs a genuinely new move.
                    if (rule.kind === "swingUp" || rule.kind === "swingDown"
                        || rule.kind === "drawSpike" || rule.kind === "stalemateBroken") {
                        ring = [{ t: now, pWin, pDraw }];
                    }
                    break;                            // at most one line per cycle
                }
            },
            roundStart() {
                const cfg = getCfg();
                const now = (typeof performance !== "undefined") ? performance.now() : Date.now();
                ring = []; holds.clear();
                cycles = 0; sentThisRound = 0; roundMin = 1; roundMax = 0;
                rounds++;
                if (rounds === 1) fire("firstRound", now, cfg);
            },
            // `won` is null for a draw / unknown.
            roundEnd(won) {
                const cfg = getCfg();
                const now = (typeof performance !== "undefined") ? performance.now() : Date.now();
                if (won === true) { winStreak++; lossStreak = 0; }
                else if (won === false) { lossStreak++; winStreak = 0; }
                else { winStreak = 0; lossStreak = 0; }   // a draw breaks both
                // A streak outranks the per-round line: it is the rarer thing
                // to have noticed, and only one of them may speak anyway.
                if (winStreak >= cfg.streakLen) fire("winStreak", now, cfg);
                else if (lossStreak >= cfg.streakLen) fire("lossStreak", now, cfg);
                else if (won === true) fire("wonRound", now, cfg);
                else if (won === false) fire("lostRound", now, cfg);
            },
            stats() {
                return { rounds, cycles, sentThisRound, winStreak, lossStreak,
                         sinceLastMs: Math.round(((typeof performance !== "undefined")
                             ? performance.now() : Date.now()) - lastSentAt),
                         roundMin: +roundMin.toFixed(3), roundMax: +roundMax.toFixed(3),
                         history: history.slice(-8) };
            },
        };
    }

    // Stateful (action, duration) decider with the FiGAR hold.
    //
    // Mirrors ppo9/policy.py ActorPolicy.act EXACTLY, and the order matters:
    //   1. hold feature = holdRemaining / max(DURATIONS)  (BEFORE decrementing)
    //   2. step the net -> this ALWAYS runs, so the recurrent state advances
    //      on every cycle whether or not the action is free
    //   3. sample both heads
    //   4. commit the sampled action/duration ONLY on a free cycle
    //   5. decrement the hold
    // Skipping step 2 on held cycles would drop most decisions out of the
    // recurrence; the memory ablation measured that as a 95.5% -> 4.5% swing.
    // `critic` is optional; when present it is stepped on EVERY cycle from its
    // own hidden state, exactly like the actor, and yields [win, draw, loss].
    function makeDecider(net, greedy, critic) {
        const MAXDUR = Math.max(...DURATIONS);
        let h = new Float32Array(net.H);
        let hc = critic ? new Float32Array(critic.H) : null;
        let holdRemaining = 0, heldAction = 0, lastDur = 0;
        // Live memory telemetry — the recurrent state is invisible otherwise,
        // and a silently-dead one costs ~95% of decisive games (measured).
        let steps = 0, resets = 0, lastDelta = 0, maxDelta = 0, lastT = 0, dtSum = 0, dtN = 0;
        return {
            decide(obs) {
                const now = (typeof performance !== "undefined") ? performance.now() : Date.now();
                if (lastT) { dtSum += now - lastT; dtN++; }
                lastT = now;

                const x = new Float32Array(net.inDim);
                x.set(obs, 0);
                x[obs.length] = holdRemaining / MAXDUR;   // 0 only when free
                const prev = h.slice();
                const logits = stepNet(net, x, h);        // memory ALWAYS steps
                let probs = null;
                if (critic) {
                    const c = stepNet(critic, x, hc);      // its own memory
                    probs = softmax(c);                    // [win, draw, loss]
                }
                let d2 = 0;
                for (let i = 0; i < h.length; i++) { const q = h[i] - prev[i]; d2 += q * q; }
                lastDelta = Math.sqrt(d2);
                if (lastDelta > maxDelta) maxDelta = lastDelta;
                steps++;

                const aL = logits.subarray(0, NUM_ACTIONS);
                const dL = logits.subarray(NUM_ACTIONS);
                const a = greedy ? argmax(aL) : sampleSoftmax(aL);
                const d = greedy ? argmax(dL) : sampleSoftmax(dL);
                const free = holdRemaining === 0;
                if (free) { heldAction = a; lastDur = DURATIONS[d]; holdRemaining = lastDur; }
                const action = heldAction;
                holdRemaining--;
                // atoms are (1,-1,-1), so V = pWin - pDraw - pLoss = 2*pWin - 1
                const value = probs ? probs[0] - probs[1] - probs[2] : 0;
                return { action, held: !free, dur: lastDur, probs, value };
            },
            // A new round: the episode boundary clears the memory, exactly as
            // ActorPolicy.reset_env does. Leaving it would bleed the previous
            // round's state into this one.
            reset() {
                h = new Float32Array(net.H);
                if (critic) hc = new Float32Array(critic.H);
                holdRemaining = 0; heldAction = 0; lastDur = 0;
                steps = 0; lastDelta = 0; maxDelta = 0; lastT = 0; dtSum = 0; dtN = 0;
                resets++;
            },
            // top.bonkai.mem() — is the hidden state actually live in-game?
            //   steps      decisions since the last round reset
            //   hNorm      |h|; 0 after a reset, should grow then hover
            //   lastDelta  |h_t - h_{t-1}| on the most recent decision; a
            //              persistent 0 with steps>1 means memory is frozen
            //   hz         MEASURED decision rate. The loop sleeps on wall
            //              clock, so a throttled tab drops decisions, and a
            //              dropped decision is a SKIPPED GRU step — the policy
            //              then sees a different memory trajectory than it
            //              trained under. Should sit near 1000/DECIDE_MS.
            mem() {
                const hz = dtN ? 1000 / (dtSum / dtN) : 0;
                let n = 0;
                for (let i = 0; i < h.length; i++) n += h[i] * h[i];
                return {
                    steps, resets, hold: holdRemaining, dur: lastDur,
                    hNorm: +Math.sqrt(n).toFixed(3),
                    lastDelta: +lastDelta.toFixed(4),
                    maxDelta: +maxDelta.toFixed(4),
                    hz: +hz.toFixed(2), expectedHz: +(1000 / DECIDE_MS).toFixed(2),
                    alive: steps > 1 && lastDelta > 0,
                    h: Array.from(h.subarray(0, 8)).map((v) => +v.toFixed(3)),
                };
            },
        };
    }

    // ---- node test hook: export the pure functions, do NOT run the browser loop
    if (!isBrowser) {
        globalThis.__play9 = {
            buildNet, stepNet, argmax, softmax, sampleSoftmax, makeDecider, indexToKeys,
            makeChatter, DEFAULT_CHAT,
            fourierExpand, matvec,
            DURATIONS, NUM_ACTIONS, STATE_DIM, AGENT_STATE_DIM,
        };
        return;
    }

    // ===== Local state (browser) ============================================
    const keyMap = new Map();
    const heavyTrack = new Map();
    let policy = null, critic = null, decider = null, chatter = null;
    let running = false, inferErrors = 0;
    // The two knobs the user asked for. `chatOn` is the live on/off switch;
    // `chatConfig` holds thresholds, cooldowns, the line lists, and `send`.
    if (isBrowser) {
        if (top.chatOn === undefined) top.chatOn = true;
        top.chatConfig = Object.assign({}, DEFAULT_CHAT, top.chatConfig || {});
    }
    const chatCfg = () => Object.assign(
        { enabled: !!(isBrowser ? top.chatOn : false) },
        isBrowser ? top.chatConfig : DEFAULT_CHAT);
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
        return out;                              // length 129
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
        if (agent.actor.type !== "mingru" || !agent.actor.gru)
            throw new Error(`not a ppo9 minGRU actor (type=${agent.actor.type}) — ppo7 saves use play7.mjs`);
        policy = buildNet(agent.actor);
        // The chat reads the CRITIC, so a checkpoint without one still plays,
        // it just never speaks.
        critic = agent.critic ? buildNet(agent.critic) : null;
        decider = makeDecider(policy, GREEDY, critic);
        chatter = makeChatter(chatCfg);
        top.bonkai.chatter = chatter;
        top.bonkai.policy = policy;
        if (policy.inDim !== AGENT_STATE_DIM)
            console.warn(`expected net input ${AGENT_STATE_DIM}, save says ${policy.inDim} — wrong model? (ppo7 uses play7.mjs)`);
        if (policy.outDim !== NUM_ACTIONS + DURATIONS.length)
            console.warn(`expected ${NUM_ACTIONS + DURATIONS.length} outputs (18 action + 7 duration), got ${policy.outDim}`);
        console.log(`ppo9 AI loaded ${policy.inDim}->${policy.outDim} (2 heads, minGRU H=${policy.H})` +
            (save.progress ? `, ELO ${Math.round(save.progress.currentRating)}` : "") +
            `. Deciding every ${ACTION_REPEAT} frames (${GREEDY ? "greedy" : "sampled"}), FiGAR holds. Stop: top.bonkai.playStop().`);

        let inRound = false;
        let lastProbs = null;
        inferErrors = 0;
        running = true;
        // One-shot confirmation that the recurrent state is live, printed a few
        // seconds into the first round. `top.bonkai.mem()` gives it on demand.
        setTimeout(() => {
            if (!decider) return;
            const m = decider.mem();
            console.log(`[memory] ${m.alive ? "LIVE" : "NOT UPDATING"} — steps=${m.steps} `
                + `|h|=${m.hNorm} d|h|=${m.lastDelta} rate=${m.hz}/${m.expectedHz}Hz`
                + (m.hz && m.hz < m.expectedHz * 0.8
                    ? "  <-- DROPPING DECISIONS: the tab is throttled, so GRU "
                      + "steps are being skipped and memory drifts from training"
                    : ""));
        }, 5000);
        while (running && isCurrent()) {
            const ids = getIds();
            if (ids) {
                if (!inRound) {
                    inRound = true;
                    roundStartMs = performance.now();
                    roundStartFrame = typeof top.getCurrentFrame === "function" ? top.getCurrentFrame() : null;
                    heavyTrack.clear();
                    decider.reset();               // new round -> free decision
                    lastProbs = null;
                    if (chatter) chatter.roundStart();
                }
                const [me, opp] = ids;
                let action = 0;
                try {
                    const out = decider.decide(buildObs(me, opp));
                    action = out.action;
                    lastProbs = out.probs;
                    if (chatter) chatter.tick(out.probs);
                }
                catch (err) {
                    // A throw means stepNet did NOT run, so this decision's GRU
                    // step is missing and the memory trajectory has desynced
                    // from what training produced. Count it rather than
                    // swallowing it silently.
                    inferErrors++;
                    if (inferErrors <= 3) console.error("inference error:", err);
                }
                playMove(action);
            } else if (inRound) {
                inRound = false;
                top.presskeys(lastMove, { up: false, down: false, left: false, right: false, heavy: false, special: false });
                lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
                lastSent = { up: 0, down: 0, left: 0, right: 0, heavy: 0 };
                heavyTrack.clear();
                // No terminal signal is exposed here, so the round's LAST
                // critic read stands in for the outcome. Deliberately
                // conservative: an ambiguous read says nothing at all.
                if (chatter) {
                    const w = lastProbs ? lastProbs[0] : null;
                    chatter.roundEnd(w === null ? null
                        : (w > 0.65 ? true : (w < 0.35 ? false : null)));
                }
                lastProbs = null;
                decider.reset();
            }
            await sleep(DECIDE_MS);
        }
        // Release the keys on ANY exit path, so a stopped or replaced instance
        // never leaves the player walking into a wall.
        try {
            top.presskeys(lastMove,
                { left: false, right: false, up: false, down: false, heavy: false, special: false });
        } catch (_) { /* nothing to release */ }
        console.log(isCurrent() ? `[play9] instance #${MY_ID} stopped.`
            : `[play9] instance #${MY_ID} retired (replaced by a newer paste).`);
    }

    // Is the hidden state actually updating in-game? Call from the console.
    top.bonkai.mem = () => (decider ? { ...decider.mem(), inferErrors } : "no decider yet");
    // top.bonkai.chat() — what the chatter is thinking, and why it is quiet.
    top.bonkai.chat = () => (chatter
        ? { on: !!top.chatOn, ...chatter.stats() } : "no chatter yet");
    top.bonkai.say = (m) => (top.chatConfig.send || defaultSend)(m);
    top.bonkai.playStop = () => { running = false; };
    top.bonkai.playRestart = () => {
        if (!isCurrent()) return console.warn(
            `[play9] instance #${MY_ID} was replaced by #${top.bonkai.instanceId}; `
            + "restart that one instead.");
        if (!running) main().catch((e) => console.error(e));
    };
    main().catch((e) => console.error("play9.mjs failed:", e));
})();
