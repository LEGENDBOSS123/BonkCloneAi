import * as tf from "@tensorflow/tfjs";
import { buildActor } from "./network.mjs";
import { actActionsBatch } from "./agent.mjs";
import { SnapshotBuffer, eloUpdate } from "./elo.mjs";

const PLAYER_KEYS = ["p1", "p2"];

// Generalized Advantage Estimation over one trajectory (may span episodes;
// `dones` mask boundaries so advantage doesn't leak across resets).
function computeGAE(rewards, values, dones, lastValue, gamma, lambda) {
    const N = rewards.length;
    const advantages = new Array(N);
    const returns = new Array(N);
    let gae = 0;
    for (let t = N - 1; t >= 0; t--) {
        const nonterminal = 1 - dones[t];
        const nextValue = t === N - 1 ? lastValue : values[t + 1];
        const delta = rewards[t] + gamma * nextValue * nonterminal - values[t];
        gae = delta + gamma * lambda * nonterminal * gae;
        advantages[t] = gae;
        returns[t] = gae + values[t];
    }
    return { advantages, returns };
}

async function modelWeightsToJSON(model) {
    // getWeights() returns the live variable tensors — read (async) but never
    // dispose them.
    return Promise.all(
        model.getWeights().map(async (w) => ({ shape: w.shape, data: Array.from(await w.data()) }))
    );
}

function setModelWeightsFromJSON(model, records) {
    const tensors = records.map((r) => tf.tensor(r.data, r.shape));
    model.setWeights(tensors);
    tensors.forEach((t) => t.dispose());
}

// Orchestrates VECTORIZED self-play PPO training over N parallel envs. Each tick
// (collectBatch) advances all envs, with a single batched forward/readback for
// all learner actions and one per distinct opponent — this amortizes the
// per-step GPU readback (the serial bottleneck) across N envs. When `rolloutSteps`
// learner transitions have accumulated (summed over envs), call runUpdate().
//
// Each env is a "slot" with its own opponent, episode counter, and two on-policy
// streams (learner always; opponent only while it IS the current model). ELO is
// tracked for rated (snapshot) matches. Two extra sample-efficiency tricks:
//   1. Opponent POV: current-model opponents' trajectories are trained on too.
//   2. Horizontal mirror: every transition is duplicated as its mirror image.
export class Trainer {
    constructor(envs, agent, config) {
        this.envs = envs;
        this.env0 = envs[0]; // rendered env; also used for the (env-agnostic) mirror
        this.agent = agent;
        this.config = config;

        this.stateSize = envs[0].stateSize;
        this.numEnvs = envs.length;
        this.learnerIndex = config.training.learnerIndex;
        this.oppIndex = this.learnerIndex === 0 ? 1 : 0;
        // Action repeat / frame-skip: the policy decides every K frames and the
        // chosen action is held for the block; rewards over the block accumulate
        // into one decision-level transition. K=1 => decide every frame.
        this.actionRepeat = Math.max(1, config.training.actionRepeat ?? 1);

        // Snapshot pool: each snapshot is a materialized actor model (so opponents
        // batch by identity without swapping weights each tick).
        this.buffer = new SnapshotBuffer(
            config.selfPlay.bufferSize,
            () => buildActor(this.stateSize, agent.actionDim, config.network)
        );
        this.currentRating = config.elo.initialRating;

        this.episodeCount = 0;
        this.updateCount = 0;
        this.recentResults = []; // learner scores (0/0.5/1) for a rolling win-rate
        // Snapshots evicted from the buffer but possibly still in use by a slot
        // mid-episode; disposed once no slot references them (see #disposeRetired).
        this.retired = [];

        this.slots = envs.map((env) => ({
            env,
            episodeSteps: 0,
            opponent: null, // { type:"current" } | { type:"snapshot", item }
            streamLearner: this.#emptyStream(),
            streamOpp: this.#emptyStream(),
            // Action-repeat state:
            stepsSinceDecision: 0, // 0 => decide this tick
            heldLearnerAction: null,
            heldOppAction: null,
            pendingLearner: null, // in-progress decision-level transition (learner)
            pendingOpp: null, // ditto for the opponent (only when opponent is current)
        }));
        for (const slot of this.slots) this.selectOpponent(slot);
    }

    #emptyStream() {
        return { states: [], actions: [], logProbs: [], values: [], rewards: [], dones: [] };
    }

    #push(stream, state, action, logProb, value, reward, done) {
        stream.states.push(state);
        stream.actions.push(action);
        stream.logProbs.push(logProb);
        stream.values.push(value);
        stream.rewards.push(reward);
        stream.dones.push(done ? 1 : 0);
    }

    #resetAllStreams() {
        for (const slot of this.slots) {
            slot.streamLearner = this.#emptyStream();
            slot.streamOpp = this.#emptyStream();
        }
    }

    #key(index) {
        return PLAYER_KEYS[index];
    }

    // Pick a fresh opponent for one slot: the live model (on-policy, trainable) or
    // a rated snapshot from the buffer.
    selectOpponent(slot) {
        const { opponentCurrentProb } = this.config.selfPlay;
        if (this.buffer.size === 0 || Math.random() < opponentCurrentProb) {
            slot.opponent = { type: "current", item: null };
        } else {
            slot.opponent = { type: "snapshot", item: this.buffer.sample() };
        }
    }

    // --- Vectorized rollout -------------------------------------------------
    // Advance every env by one frame. Only slots whose action-repeat block has
    // expired query the policy (holding actions otherwise); a decision-level
    // transition is stored when a block completes or the episode ends. Returns an
    // array of episode-end infos.
    async collectBatch() {
        const li = this.learnerIndex;
        const oi = this.oppIndex;
        const N = this.slots.length;
        const K = this.actionRepeat;
        const opStochastic = this.config.selfPlay.opponentStochastic;

        // 1. Slots that need a fresh decision this frame (start of a new block).
        const deciding = [];
        for (let i = 0; i < N; i++) if (this.slots[i].stepsSinceDecision === 0) deciding.push(i);

        if (deciding.length) {
            const lStates = deciding.map((i) => this.slots[i].env.state(li));
            const oStates = deciding.map((i) => this.slots[i].env.state(oi));
            const lOut = await this.agent.actBatch(lStates, false);

            // Opponent decisions for the deciding slots, grouped by identity.
            const oppAct = new Array(deciding.length);
            const oppLp = new Array(deciding.length);
            const oppV = new Array(deciding.length);
            const curPos = []; // positions (within `deciding`) whose opponent is current
            deciding.forEach((i, k) => { if (this.slots[i].opponent.type === "current") curPos.push(k); });
            if (curPos.length) {
                const out = await this.agent.actBatch(curPos.map((k) => oStates[k]), false);
                curPos.forEach((k, j) => { oppAct[k] = out.actions[j]; oppLp[k] = out.logProbs[j]; oppV[k] = out.values[j]; });
            }
            const snapGroups = new Map(); // snapshot item -> [positions within `deciding`]
            deciding.forEach((i, k) => {
                const o = this.slots[i].opponent;
                if (o.type !== "snapshot") return;
                if (!snapGroups.has(o.item)) snapGroups.set(o.item, []);
                snapGroups.get(o.item).push(k);
            });
            for (const [item, ks] of snapGroups) {
                const acts = await actActionsBatch(item.actor, ks.map((k) => oStates[k]), !opStochastic);
                ks.forEach((k, j) => (oppAct[k] = acts[j]));
            }

            // Commit each decision: set the held action and open a pending transition.
            deciding.forEach((i, k) => {
                const slot = this.slots[i];
                slot.heldLearnerAction = lOut.actions[k];
                slot.pendingLearner = { state: lStates[k], action: lOut.actions[k], logProb: lOut.logProbs[k], value: lOut.values[k], reward: 0 };
                slot.heldOppAction = oppAct[k];
                slot.pendingOpp = slot.opponent.type === "current"
                    ? { state: oStates[k], action: oppAct[k], logProb: oppLp[k], value: oppV[k], reward: 0 }
                    : null;
            });
        }

        // 2. Step every env with its held action; accumulate reward; finalize the
        //    decision-level transition when the block ends (K frames) or on done.
        const infos = [];
        for (let i = 0; i < N; i++) {
            const slot = this.slots[i];
            const inputs = {};
            inputs[this.#key(li)] = slot.heldLearnerAction;
            inputs[this.#key(oi)] = slot.heldOppAction;
            const res = slot.env.step(inputs);

            slot.episodeSteps++;
            slot.stepsSinceDecision++;
            slot.pendingLearner.reward += res.reward[this.#key(li)];
            if (slot.pendingOpp) slot.pendingOpp.reward += res.reward[this.#key(oi)];

            const timeout = slot.episodeSteps >= this.config.env.maxEpisodeSteps;
            const done = res.done || timeout;

            if (slot.stepsSinceDecision >= K || done) {
                const pl = slot.pendingLearner;
                this.#push(slot.streamLearner, pl.state, pl.action, pl.logProb, pl.value, pl.reward, done);
                if (slot.pendingOpp) {
                    const po = slot.pendingOpp;
                    this.#push(slot.streamOpp, po.state, po.action, po.logProb, po.value, po.reward, done);
                }
                slot.pendingLearner = null;
                slot.pendingOpp = null;
                slot.stepsSinceDecision = 0; // next frame starts a fresh decision block
                if (done) infos.push(this.#finishEpisode(slot, res, timeout));
            }
        }
        this.#disposeRetired();
        return infos;
    }

    #finishEpisode(slot, res, timeout) {
        const deadL = res.dead[this.#key(this.learnerIndex)];
        const deadO = res.dead[this.#key(this.oppIndex)];

        // Learner score: win = opponent died & learner survived; draw = both died
        // or timeout; else loss.
        let score;
        if (timeout || (deadL && deadO)) score = 0.5;
        else if (deadO && !deadL) score = 1;
        else score = 0;

        // ELO updates only for rated (snapshot) matches.
        if (slot.opponent.type === "snapshot") {
            const [rNew, sNew] = eloUpdate(this.currentRating, slot.opponent.item.rating, score, this.config.elo.kFactor);
            this.currentRating = rNew;
            slot.opponent.item.rating = sNew;
        }

        this.episodeCount++;
        this.recentResults.push(score);
        if (this.recentResults.length > 100) this.recentResults.shift();

        // Snapshot the current actor into the pool every N (global) episodes.
        if (this.episodeCount % this.config.selfPlay.snapshotIntervalEpisodes === 0) {
            const evicted = this.buffer.addFromWeights(this.agent.actor.getWeights(), this.currentRating, this.episodeCount);
            if (evicted) this.retired.push(evicted);
        }

        slot.env.reset();
        slot.episodeSteps = 0;
        this.selectOpponent(slot);
        return { score, rating: this.currentRating, episode: this.episodeCount };
    }

    // Dispose evicted snapshots that no slot is using anymore (a slot may still be
    // mid-episode against a just-evicted snapshot; keep it until that finishes).
    #disposeRetired() {
        if (!this.retired.length) return;
        const inUse = new Set();
        for (const slot of this.slots) if (slot.opponent.type === "snapshot") inUse.add(slot.opponent.item);
        this.retired = this.retired.filter((item) => {
            if (inUse.has(item)) return true;
            item.actor.dispose();
            return false;
        });
    }

    // Total learner transitions collected across all envs / opponent bonus data.
    learnerSteps() {
        return this.slots.reduce((a, s) => a + s.streamLearner.states.length, 0);
    }
    opponentSteps() {
        return this.slots.reduce((a, s) => a + s.streamOpp.states.length, 0);
    }

    needsUpdate() {
        return this.learnerSteps() >= this.config.ppo.rolloutSteps;
    }

    // GAE (per env-stream) + horizontal-mirror augmentation + PPO update, clear.
    async runUpdate() {
        const { gamma, gaeLambda } = this.config.ppo;
        const D = this.stateSize;
        const A = this.agent.actionDim;

        // Every env contributes a learner trajectory and (when it played the live
        // model) an opponent trajectory. `pending` is the in-flight (mid-block)
        // transition, whose value is exactly V(next decision state) for bootstrap.
        const streams = [];
        for (const slot of this.slots) {
            streams.push({ stream: slot.streamLearner, index: this.learnerIndex, env: slot.env, pending: slot.pendingLearner });
            streams.push({ stream: slot.streamOpp, index: this.oppIndex, env: slot.env, pending: slot.pendingOpp });
        }

        // Bootstrap: a stream needs V(next state) only if its last transition is
        // non-terminal. If a block is mid-flight, its pending transition's value is
        // that V; otherwise the env sits on the next decision state, so batch it.
        const bootReqs = [];
        for (const st of streams) {
            const n = st.stream.states.length;
            st.lastValue = 0;
            if (n > 0 && !st.stream.dones[n - 1]) {
                if (st.pending) st.lastValue = st.pending.value;
                else bootReqs.push({ st, state: st.env.state(st.index) });
            }
        }
        if (bootReqs.length) {
            const vals = await this.agent.valueBatch(bootReqs.map((b) => b.state));
            bootReqs.forEach((b, k) => (b.st.lastValue = vals[k]));
        }

        // GAE per stream, then flatten to rows.
        const rows = { states: [], actions: [], logProbs: [], values: [], advantages: [], returns: [] };
        for (const st of streams) {
            const stream = st.stream;
            const n = stream.states.length;
            if (n === 0) continue;
            const { advantages, returns } = computeGAE(stream.rewards, stream.values, stream.dones, st.lastValue, gamma, gaeLambda);
            for (let i = 0; i < n; i++) {
                rows.states.push(stream.states[i]);
                rows.actions.push(stream.actions[i]);
                rows.logProbs.push(stream.logProbs[i]);
                rows.values.push(stream.values[i]);
                rows.advantages.push(advantages[i]);
                rows.returns.push(returns[i]);
            }
        }

        // Horizontal-mirror augmentation (config.ppo.mirrorAugment): each row also
        // contributes its mirror image, reusing value/advantage/return/log-prob.
        const mirror = this.config.ppo.mirrorAugment;
        const M = rows.states.length;
        const total = mirror ? M * 2 : M;
        const states = new Float32Array(total * D);
        const actions = new Float32Array(total * A);
        const logProbs = new Float32Array(total);
        const values = new Float32Array(total);
        const advantages = new Float32Array(total);
        const returns = new Float32Array(total);
        for (let i = 0; i < M; i++) {
            states.set(rows.states[i], i * D);
            actions.set(rows.actions[i], i * A);
            logProbs[i] = rows.logProbs[i];
            values[i] = rows.values[i];
            advantages[i] = rows.advantages[i];
            returns[i] = rows.returns[i];
            if (mirror) {
                const j = M + i;
                states.set(this.env0.mirrorState(rows.states[i]), j * D);
                actions.set(this.env0.mirrorAction(rows.actions[i]), j * A);
                logProbs[j] = rows.logProbs[i];
                values[j] = rows.values[i];
                advantages[j] = rows.advantages[i];
                returns[j] = rows.returns[i];
            }
        }

        const stats = await this.agent.update({ states, actions, logProbs, values, returns, advantages });
        this.updateCount++;
        this.#resetAllStreams();
        return stats;
    }

    // Rolling win-rate (counting draws as 0.5) over the last ~100 episodes.
    winRate() {
        if (!this.recentResults.length) return 0;
        return this.recentResults.reduce((a, b) => a + b, 0) / this.recentResults.length;
    }

    // How many slots are currently facing the live model vs a snapshot.
    opponentMix() {
        let current = 0;
        for (const s of this.slots) if (s.opponent.type === "current") current++;
        return { current, snapshot: this.slots.length - current };
    }

    // --- Save / load --------------------------------------------------------
    async serialize() {
        const [actor, critic, snapshots] = await Promise.all([
            modelWeightsToJSON(this.agent.actor),
            modelWeightsToJSON(this.agent.critic),
            this.buffer.serialize(),
        ]);
        return {
            version: 1,
            savedAt: new Date().toISOString(),
            progress: {
                episodeCount: this.episodeCount,
                updateCount: this.updateCount,
                currentRating: this.currentRating,
            },
            actor,
            critic,
            snapshots,
        };
    }

    loadState(obj) {
        setModelWeightsFromJSON(this.agent.actor, obj.actor);
        setModelWeightsFromJSON(this.agent.critic, obj.critic);
        // Slots are reselected below, so nothing references the old/retired
        // snapshots anymore — dispose them before loading the new buffer.
        for (const item of this.retired) item.actor.dispose();
        this.retired = [];
        this.buffer.load(obj.snapshots || []);
        this.episodeCount = obj.progress?.episodeCount ?? 0;
        this.updateCount = obj.progress?.updateCount ?? 0;
        this.currentRating = obj.progress?.currentRating ?? this.config.elo.initialRating;
        // Restart every env's episode/streams and pick opponents from the restored buffer.
        for (const slot of this.slots) {
            slot.env.reset();
            slot.episodeSteps = 0;
            slot.streamLearner = this.#emptyStream();
            slot.streamOpp = this.#emptyStream();
            slot.stepsSinceDecision = 0;
            slot.heldLearnerAction = null;
            slot.heldOppAction = null;
            slot.pendingLearner = null;
            slot.pendingOpp = null;
            this.selectOpponent(slot);
        }
    }
}
