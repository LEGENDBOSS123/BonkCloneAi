import * as tf from "@tensorflow/tfjs";
import { buildActor, buildCritic } from "./network.mjs";

// Yield to the event loop without depending on requestAnimationFrame — unlike
// tf.nextFrame(), this still resolves when the tab is in the background (where
// rAF is paused), so a PPO update can't hang a hidden tab.
const macroYield = () => new Promise((resolve) => setTimeout(resolve, 0));

// PPO agent with a multi-binary (independent Bernoulli) policy: the actor emits
// `actionDim` logits, one per key, and each action bit is sampled independently.
// Actor and critic are separate networks (no shared backbone) with their own Adam
// optimizers. Training uses the clipped PPO objective with an entropy bonus, a
// (clipped) value loss, and global-norm gradient clipping per network.

// log(sigmoid(z)) = -softplus(-z); numerically stable.
function logSigmoid(z) {
    return tf.softplus(z.neg()).neg();
}

// Sum over action dims of the per-bit Bernoulli log-prob. logits/actions: [B, A].
function bernoulliLogProb(logits, actions) {
    const lp1 = logSigmoid(logits); // log P(bit = 1)
    const lp0 = logSigmoid(logits.neg()); // log P(bit = 0)
    const term = actions.mul(lp1).add(tf.scalar(1).sub(actions).mul(lp0));
    return term.sum(1); // [B]
}

// Summed Bernoulli entropy per sample. logits: [B, A] -> [B].
function bernoulliEntropy(logits) {
    const p = tf.sigmoid(logits);
    const lp1 = logSigmoid(logits);
    const lp0 = logSigmoid(logits.neg());
    const h = p.mul(lp1).add(tf.scalar(1).sub(p).mul(lp0)).neg();
    return h.sum(1); // [B]
}

// Sample a multi-binary action tensor from logits. [B, A] -> [B, A] of 0/1.
function sampleBernoulli(logits, deterministic) {
    const p = tf.sigmoid(logits);
    if (deterministic) return p.greater(0.5).cast("float32");
    return tf.randomUniform(p.shape).less(p).cast("float32");
}

// Sample an action (JS number[]) from an arbitrary actor model — used to drive
// the frozen self-play opponent without needing log-probs or values.
export async function actFromActor(actorModel, stateArr, deterministic = false) {
    const action = tf.tidy(() => sampleBernoulli(actorModel.apply(tf.tensor2d([stateArr])), deterministic));
    const arr = await action.data(); // async GPU->CPU read (fast on WebGPU)
    action.dispose();
    return Array.from(arr);
}

// Batched version of actFromActor: states is number[N][A_in] -> actions number[N][A].
export async function actActionsBatch(actorModel, states, deterministic = false) {
    const action = tf.tidy(() => sampleBernoulli(actorModel.apply(tf.tensor2d(states)), deterministic));
    const data = await action.data();
    const A = action.shape[1];
    action.dispose();
    const out = [];
    for (let i = 0; i < states.length; i++) out.push(Array.from(data.subarray(i * A, i * A + A)));
    return out;
}

// The trainable tf.Variables behind a model's layers (for per-network grads).
function trainableVars(model) {
    return model.trainableWeights.map((w) => w.val);
}

// Scale a gradient map so its global L2 norm is <= maxNorm. Runs inside a tidy.
// The per-grad squared sums are reduced pairwise with add() rather than tf.addN:
// addN builds a single kernel taking every input as a storage buffer, which on
// WebGPU can exceed the 8-storage-buffer-per-stage limit (a net with 8 weight
// tensors -> 8 inputs + 1 output = 9). Pairwise add is 2 inputs per kernel.
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

export class PPOAgent {
    constructor(stateDim, actionDim, config) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.cfg = config.ppo;
        const netCfg = config.network;

        this.actor = buildActor(stateDim, actionDim, netCfg);
        this.critic = buildCritic(stateDim, netCfg);
        this.actorOpt = tf.train.adam(this.cfg.actorLr);
        this.criticOpt = tf.train.adam(this.cfg.criticLr);
        this.actorVars = trainableVars(this.actor);
        this.criticVars = trainableVars(this.critic);
    }

    // Sample one action for a single state. Returns { action:number[A], logProb, value }.
    // Async so the GPU->CPU readback uses the non-blocking .data() path (WebGPU).
    async act(stateArr, deterministic = false) {
        // Compute inside a tidy but keep the three tensors we need to read; the
        // intermediates are freed on tidy exit. We read them async, then dispose.
        const { action, logProb, value } = tf.tidy(() => {
            const st = tf.tensor2d([stateArr]);
            const logits = this.actor.apply(st);
            const a = sampleBernoulli(logits, deterministic);
            return {
                action: a,
                logProb: bernoulliLogProb(logits, a),
                value: this.critic.apply(st).reshape([-1]),
            };
        });
        const [actionArr, logProbArr, valueArr] = await Promise.all([action.data(), logProb.data(), value.data()]);
        tf.dispose([action, logProb, value]);
        return { action: Array.from(actionArr), logProb: logProbArr[0], value: valueArr[0] };
    }

    // Critic value of a single state (for GAE bootstrap).
    async valueOf(stateArr) {
        const v = tf.tidy(() => this.critic.apply(tf.tensor2d([stateArr])).reshape([-1]));
        const arr = await v.data();
        v.dispose();
        return arr[0];
    }

    // Batched act: states is number[N][D] -> { actions:number[N][A], logProbs:number[N], values:number[N] }.
    // One forward + one GPU->CPU readback for all N — this is the vectorized-env win.
    async actBatch(states, deterministic = false) {
        const N = states.length;
        const A = this.actionDim;
        const { action, logProb, value } = tf.tidy(() => {
            const st = tf.tensor2d(states);
            const logits = this.actor.apply(st);
            const a = sampleBernoulli(logits, deterministic);
            return { action: a, logProb: bernoulliLogProb(logits, a), value: this.critic.apply(st).reshape([-1]) };
        });
        const [aData, lpData, vData] = await Promise.all([action.data(), logProb.data(), value.data()]);
        tf.dispose([action, logProb, value]);
        const actions = [];
        for (let i = 0; i < N; i++) actions.push(Array.from(aData.subarray(i * A, i * A + A)));
        return { actions, logProbs: Array.from(lpData), values: Array.from(vData) };
    }

    // Batched critic values: states number[N][D] -> number[N] (for GAE bootstrap).
    async valueBatch(states) {
        if (!states.length) return [];
        const v = tf.tidy(() => this.critic.apply(tf.tensor2d(states)).reshape([-1]));
        const arr = await v.data();
        v.dispose();
        return Array.from(arr);
    }

    // Independent, frozen copies of the actor weights (caller owns/disposes them).
    // getWeights() returns the LIVE variable tensors, so we must clone — keeping
    // them directly would both alias the mutating variables and, when the caller
    // later disposes them, dispose the model's real weights.
    cloneActorWeights() {
        return this.actor.getWeights().map((w) => tf.keep(w.clone()));
    }

    // One PPO update over a rollout. `batch` holds flat typed arrays:
    //   states  : Float32Array [N*stateDim]   actions: Float32Array [N*actionDim]
    //   logProbs, values, returns, advantages : Float32Array [N]
    async update(batch) {
        const { epochs, minibatchSize, clipEpsilon, entropyCoef, valueCoef, maxGradNorm } = this.cfg;
        const N = batch.returns.length;
        const D = this.stateDim;
        const A = this.actionDim;

        // --- perf instrumentation (temporary): where does the update time go? ---
        const _t0 = performance.now();
        const _numSteps = Math.ceil(N / minibatchSize) * epochs;

        const statesT = tf.tensor2d(batch.states, [N, D]);
        const actionsT = tf.tensor2d(batch.actions, [N, A]);
        const oldLogT = tf.tensor1d(batch.logProbs);
        const oldValT = tf.tensor1d(batch.values);
        const returnsT = tf.tensor1d(batch.returns);
        let advT = tf.tensor1d(batch.advantages);
        if (this.cfg.normalizeAdvantages) {
            // tidy so the intermediates (mean, variance, sub/sqrt/add) are freed —
            // only the normalized result escapes. Without this, 5 tensors leak per
            // update (each pinning a GPU buffer on the WebGPU backend).
            const normed = tf.tidy(() => {
                const { mean, variance } = tf.moments(advT);
                return advT.sub(mean).div(tf.sqrt(variance).add(1e-8));
            });
            advT.dispose();
            advT = tf.keep(normed);
        }

        const idx = Array.from({ length: N }, (_, i) => i);
        // Per-minibatch loss scalars are kept and read asynchronously after the
        // loop (never dataSync in the training hot path on WebGPU).
        const lossTensors = []; // { a, c, e } kept scalars

        for (let e = 0; e < epochs; e++) {
            // Fisher–Yates shuffle for minibatch ordering.
            for (let i = N - 1; i > 0; i--) {
                const j = (Math.random() * (i + 1)) | 0;
                [idx[i], idx[j]] = [idx[j], idx[i]];
            }

            for (let s = 0; s < N; s += minibatchSize) {
                const mbIdx = idx.slice(s, s + minibatchSize);
                tf.tidy(() => {
                    const idxT = tf.tensor1d(mbIdx, "int32");
                    const mbStates = statesT.gather(idxT);
                    const mbActions = actionsT.gather(idxT);
                    const mbOldLog = oldLogT.gather(idxT);
                    const mbAdv = advT.gather(idxT);
                    const mbRet = returnsT.gather(idxT);
                    const mbOldVal = oldValT.gather(idxT);

                    // --- Actor: clipped surrogate - entropy bonus ---------------
                    // Entropy is captured from THIS forward (kept) so logging costs
                    // no extra forward pass.
                    let entKept = null;
                    const { value: aLoss, grads: aGrads } = tf.variableGrads(() => {
                        const logits = this.actor.apply(mbStates, { training: true });
                        const newLog = bernoulliLogProb(logits, mbActions);
                        const ratio = newLog.sub(mbOldLog).exp();
                        const s1 = ratio.mul(mbAdv);
                        const s2 = ratio.clipByValue(1 - clipEpsilon, 1 + clipEpsilon).mul(mbAdv);
                        const policyLoss = tf.minimum(s1, s2).mean().neg();
                        const entropy = bernoulliEntropy(logits).mean();
                        entKept = tf.keep(entropy);
                        return policyLoss.sub(entropy.mul(entropyCoef));
                    }, this.actorVars);
                    this.actorOpt.applyGradients(clipGlobalNorm(aGrads, maxGradNorm));

                    // --- Critic: (clipped) value loss --------------------------
                    const { value: cLoss, grads: cGrads } = tf.variableGrads(() => {
                        const vpred = this.critic.apply(mbStates, { training: true }).reshape([-1]);
                        let vLoss;
                        if (this.cfg.clipValueLoss) {
                            const vClipped = mbOldVal.add(
                                vpred.sub(mbOldVal).clipByValue(-clipEpsilon, clipEpsilon)
                            );
                            vLoss = tf.maximum(vpred.sub(mbRet).square(), vClipped.sub(mbRet).square()).mean();
                        } else {
                            vLoss = vpred.sub(mbRet).square().mean();
                        }
                        return vLoss.mul(valueCoef);
                    }, this.criticVars);
                    this.criticOpt.applyGradients(clipGlobalNorm(cGrads, maxGradNorm));

                    lossTensors.push({ a: tf.keep(aLoss), c: tf.keep(cLoss), e: entKept });
                });
            }
            // Yield between epochs so the UI/render can breathe (background-safe).
            await macroYield();
        }

        statesT.dispose();
        actionsT.dispose();
        oldLogT.dispose();
        oldValT.dispose();
        returnsT.dispose();
        advT.dispose();

        // Async-read the accumulated loss stats, then dispose them.
        let actorLoss = 0, criticLoss = 0, entropy = 0;
        for (const t of lossTensors) {
            const [a, c, e] = await Promise.all([t.a.data(), t.c.data(), t.e.data()]);
            actorLoss += a[0];
            criticLoss += c[0];
            entropy += e[0];
            tf.dispose([t.a, t.c, t.e]);
        }
        const n = Math.max(1, lossTensors.length);

        const _ms = performance.now() - _t0;
        console.log(
            `[update] backend=${tf.getBackend()} N=${N} steps=${_numSteps} ` +
            `total=${_ms.toFixed(0)}ms per-step=${(_ms / Math.max(1, _numSteps)).toFixed(1)}ms ` +
            `liveTensors=${tf.memory().numTensors}`
        );

        return { actorLoss: actorLoss / n, criticLoss: criticLoss / n, entropy: entropy / n };
    }
}
