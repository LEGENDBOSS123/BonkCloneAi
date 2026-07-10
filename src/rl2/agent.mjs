import * as tf from "@tensorflow/tfjs";
import { buildQNet, buildPolicyNet, buildAvgNet, weightsToJSON, weightsFromJSON } from "./networks.mjs";

// NFSP agent = discrete-SAC best response + supervised average policy.
//
//   BR (SAC)  : twin Q nets (+ Polyak targets), softmax policy, auto-tuned
//               entropy temperature alpha. Trained off-policy from the replay
//               buffer with n-step targets. Exact expectations over the 18
//               discrete actions (no sampling inside the losses).
//   AVG       : softmax classifier trained by cross-entropy on the reservoir of
//               historical BR (state, action) pairs. This is the deploy policy.
//
// GPU hygiene (hard-won in rl1): losses are only read back every `statsEvery`
// steps (async), grad-norm clipping reduces pairwise (WebGPU storage-buffer
// limit), and every step runs inside tidies so nothing leaks.

function trainableVars(model) {
    return model.trainableWeights.map((w) => w.val);
}

// Scale a gradient map so its global L2 norm is <= maxNorm (pairwise reduce —
// see rl1 agent for why not tf.addN on WebGPU).
function clipGlobalNorm(grads, maxNorm) {
    if (!maxNorm) return grads;
    const sq = Object.values(grads).map((g) => g.square().sum());
    const sumSq = sq.reduce((acc, t) => acc.add(t));
    const globalNorm = tf.sqrt(sumSq);
    const factor = tf.minimum(tf.scalar(1), tf.scalar(maxNorm).div(globalNorm.add(1e-6)));
    const out = {};
    for (const k of Object.keys(grads)) out[k] = grads[k].mul(factor);
    return out;
}

// Sample one action index from softmax(logits row) on the CPU.
function sampleRow(data, off, A) {
    let max = -Infinity;
    for (let a = 0; a < A; a++) if (data[off + a] > max) max = data[off + a];
    let sum = 0;
    const probs = new Array(A);
    for (let a = 0; a < A; a++) {
        probs[a] = Math.exp(data[off + a] - max);
        sum += probs[a];
    }
    let r = Math.random() * sum;
    for (let a = 0; a < A; a++) {
        r -= probs[a];
        if (r <= 0) return a;
    }
    return A - 1;
}

function argmaxRow(data, off, A) {
    let best = 0;
    let bestVal = -Infinity;
    for (let a = 0; a < A; a++) if (data[off + a] > bestVal) { bestVal = data[off + a]; best = a; }
    return best;
}

export class NFSPAgent {
    constructor(stateDim, config) {
        this.stateDim = stateDim;
        this.numActions = config.numActions;
        this.cfg = config;
        const net = config.network;
        const sac = config.sac;

        this.q1 = buildQNet(stateDim, this.numActions, net);
        this.q2 = buildQNet(stateDim, this.numActions, net);
        this.q1t = buildQNet(stateDim, this.numActions, net);
        this.q2t = buildQNet(stateDim, this.numActions, net);
        this.q1t.setWeights(this.q1.getWeights());
        this.q2t.setWeights(this.q2.getWeights());
        this.policy = buildPolicyNet(stateDim, this.numActions, net);
        this.avg = buildAvgNet(stateDim, this.numActions, net);

        this.qOpt = tf.train.adam(sac.qLr);
        this.policyOpt = tf.train.adam(sac.policyLr);
        this.alphaOpt = tf.train.adam(sac.alphaLr);
        this.avgOpt = tf.train.adam(config.avg.lr);

        this.logAlpha = tf.variable(tf.scalar(Math.log(sac.initAlpha)), true);
        this.targetEntropy = sac.targetEntropyRatio * Math.log(this.numActions);

        this.qVars = [...trainableVars(this.q1), ...trainableVars(this.q2)];
        this.q1Pairs = this.#targetPairs(this.q1, this.q1t);
        this.q2Pairs = this.#targetPairs(this.q2, this.q2t);
        this.policyVars = trainableVars(this.policy);
        this.avgVars = trainableVars(this.avg);

        this.sacSteps = 0;
        this.slSteps = 0;
        // Rolling training stats, refreshed every statsEvery steps (async read).
        this.stats = { qLoss: 0, policyLoss: 0, avgLoss: 0, entropy: 0, alpha: sac.initAlpha };
    }

    #targetPairs(src, dst) {
        const s = trainableVars(src);
        const d = trainableVars(dst);
        return s.map((v, i) => [v, d[i]]);
    }

    get alpha() {
        return this.stats.alpha;
    }

    // --- acting ----------------------------------------------------------------
    // states: Float32Array[N] (each stateDim long). which: "br" | "avg" | a model.
    // Returns Int32Array of N action indices.
    async actBatch(which, states, greedy = false) {
        const model = which === "br" ? this.policy : which === "avg" ? this.avg : which;
        const N = states.length;
        const D = this.stateDim;
        const A = this.numActions;
        const flat = new Float32Array(N * D);
        for (let i = 0; i < N; i++) flat.set(states[i], i * D);
        const logits = tf.tidy(() => model.apply(tf.tensor2d(flat, [N, D])));
        const data = await logits.data();
        logits.dispose();
        const out = new Int32Array(N);
        for (let i = 0; i < N; i++) out[i] = greedy ? argmaxRow(data, i * A, A) : sampleRow(data, i * A, A);
        return out;
    }

    // --- SAC best-response step --------------------------------------------------
    // batch: { states, actions, rewards, nextStates, dones, gammaN, n } (typed).
    sacStep(batch) {
        const { n } = batch;
        const D = this.stateDim;
        const A = this.numActions;
        const { maxGradNorm } = this.cfg.sac;
        const wantStats = this.sacSteps % this.cfg.training.statsEvery === 0;
        const kept = wantStats ? {} : null;

        tf.tidy(() => {
            const st = tf.tensor2d(batch.states, [n, D]);
            const st2 = tf.tensor2d(batch.nextStates, [n, D]);
            const actIdx = tf.tensor1d(batch.actions, "int32");
            const aHot = tf.oneHot(actIdx, A).cast("float32");
            const rew = tf.tensor1d(batch.rewards);
            const notDone = tf.scalar(1).sub(tf.tensor1d(batch.dones));
            const gammaN = tf.tensor1d(batch.gammaN);
            const alpha = this.logAlpha.exp();

            // Entropy-regularized n-step target (exact expectation over actions):
            //   y = r + gamma^n * (1-done) * sum_a' pi(a'|s') [minQt(s',a') - alpha log pi]
            const y = tf.tidy(() => {
                const logits2 = this.policy.apply(st2);
                const logp2 = tf.logSoftmax(logits2);
                const p2 = tf.softmax(logits2);
                const minQt = tf.minimum(this.q1t.apply(st2), this.q2t.apply(st2));
                const v2 = p2.mul(minQt.sub(logp2.mul(alpha))).sum(1);
                return rew.add(gammaN.mul(notDone).mul(v2));
            });

            // --- twin-Q update ------------------------------------------------
            const { value: qLoss, grads: qGrads } = tf.variableGrads(() => {
                const qa1 = this.q1.apply(st, { training: true }).mul(aHot).sum(1);
                const qa2 = this.q2.apply(st, { training: true }).mul(aHot).sum(1);
                return qa1.sub(y).square().mean().add(qa2.sub(y).square().mean());
            }, this.qVars);
            this.qOpt.applyGradients(clipGlobalNorm(qGrads, maxGradNorm));

            // --- policy update --------------------------------------------------
            let entropyKept = null;
            const { value: pLoss, grads: pGrads } = tf.variableGrads(() => {
                const logits = this.policy.apply(st, { training: true });
                const logp = tf.logSoftmax(logits);
                const p = tf.softmax(logits);
                const minQ = tf.minimum(this.q1.apply(st), this.q2.apply(st));
                const entropy = p.mul(logp).sum(1).neg().mean();
                entropyKept = tf.keep(entropy);
                return p.mul(logp.mul(alpha).sub(minQ)).sum(1).mean();
            }, this.policyVars);
            this.policyOpt.applyGradients(clipGlobalNorm(pGrads, maxGradNorm));

            // --- temperature update: alpha moves toward the entropy target -----
            const entropyConst = entropyKept.clone(); // constant wrt logAlpha
            const { grads: aGrads } = tf.variableGrads(
                () => this.logAlpha.exp().mul(entropyConst.sub(this.targetEntropy)),
                [this.logAlpha]
            );
            this.alphaOpt.applyGradients(aGrads);
            // Clamp: alpha must not decay to ~0 (see config.sac.minAlpha).
            this.logAlpha.assign(tf.maximum(this.logAlpha, tf.scalar(Math.log(this.cfg.sac.minAlpha))));

            // --- Polyak-average the target nets (thinned; see config) -----------
            if (this.sacSteps % (this.cfg.sac.polyakEvery ?? 1) === 0) {
                const tau = this.cfg.sac.tau;
                for (const pairs of [this.q1Pairs, this.q2Pairs]) {
                    for (const [src, dst] of pairs) dst.assign(dst.mul(1 - tau).add(src.mul(tau)));
                }
            }

            if (wantStats) {
                kept.q = tf.keep(qLoss.clone());
                kept.p = tf.keep(pLoss.clone());
                kept.h = tf.keep(entropyKept.clone());
                kept.a = tf.keep(this.logAlpha.exp().clone());
            }
            entropyKept.dispose();
        });

        this.sacSteps++;
        if (kept) this.#readStats(kept);
    }

    // Async, out-of-band stats readback — never stalls the training path.
    async #readStats(kept) {
        try {
            const [q, p, h, a] = await Promise.all([kept.q.data(), kept.p.data(), kept.h.data(), kept.a.data()]);
            this.stats.qLoss = q[0];
            this.stats.policyLoss = p[0];
            this.stats.entropy = h[0];
            this.stats.alpha = a[0];
        } finally {
            tf.dispose([kept.q, kept.p, kept.h, kept.a]);
        }
    }

    // --- average-policy (supervised) step ---------------------------------------
    // batch: { states, actions, n } from the reservoir buffer.
    slStep(batch) {
        const { n } = batch;
        const D = this.stateDim;
        const A = this.numActions;
        const wantStats = this.slSteps % this.cfg.training.statsEvery === 0;
        let keptLoss = null;

        tf.tidy(() => {
            const st = tf.tensor2d(batch.states, [n, D]);
            const aHot = tf.oneHot(tf.tensor1d(batch.actions, "int32"), A).cast("float32");
            const { value: loss, grads } = tf.variableGrads(() => {
                const logp = tf.logSoftmax(this.avg.apply(st, { training: true }));
                return aHot.mul(logp).sum(1).neg().mean(); // cross-entropy
            }, this.avgVars);
            this.avgOpt.applyGradients(clipGlobalNorm(grads, this.cfg.sac.maxGradNorm));
            if (wantStats) keptLoss = tf.keep(loss.clone());
        });

        this.slSteps++;
        if (keptLoss) {
            keptLoss.data().then((v) => {
                this.stats.avgLoss = v[0];
                keptLoss.dispose();
            });
        }
    }

    // --- save / load ---------------------------------------------------------------
    async serialize() {
        // Clone before awaiting anything — same live-variable race as
        // weightsToJSON (the alpha optimizer assigns to logAlpha every step).
        const logAlphaClone = this.logAlpha.clone();
        const logAlpha = (await logAlphaClone.data())[0];
        logAlphaClone.dispose();
        return {
            q1: await weightsToJSON(this.q1),
            q2: await weightsToJSON(this.q2),
            q1t: await weightsToJSON(this.q1t),
            q2t: await weightsToJSON(this.q2t),
            policy: await weightsToJSON(this.policy),
            avg: await weightsToJSON(this.avg),
            logAlpha,
            sacSteps: this.sacSteps,
            slSteps: this.slSteps,
        };
    }

    loadState(obj) {
        weightsFromJSON(this.q1, obj.q1);
        weightsFromJSON(this.q2, obj.q2);
        weightsFromJSON(this.q1t, obj.q1t);
        weightsFromJSON(this.q2t, obj.q2t);
        weightsFromJSON(this.policy, obj.policy);
        weightsFromJSON(this.avg, obj.avg);
        if (typeof obj.logAlpha === "number") {
            tf.tidy(() => this.logAlpha.assign(tf.scalar(obj.logAlpha)));
        }
        this.sacSteps = obj.sacSteps ?? 0;
        this.slSteps = obj.slSteps ?? 0;
    }
}
