import { SnapshotBuffer, eloUpdate } from "../rl/elo.mjs";
import { buildAvgNet } from "./networks.mjs";
import { ReplayBuffer, ReservoirBuffer } from "./buffers.mjs";
import { mirrorObs, mirrorActionIndex } from "./lag_env.mjs";

// Vectorized NFSP self-play over N parallel LagEnvs.
//
// Per episode, each seat independently plays the BEST RESPONSE (prob eta) or the
// AVERAGE policy. BR seats log (state, action) into the SL reservoir; every
// seat's decision-level transitions go into the RL replay (off-policy — data
// from any behavior policy is valid for the SAC learner). A small fraction of
// episodes are rated eval matches: current AVG vs a frozen AVG snapshot (ELO).
//
// Gradient steps are interleaved with collection (a few small batches per env
// tick) instead of big blocking updates, so the tab stays responsive and the
// learner tracks the data closely.
export class Trainer {
    constructor(lagEnvs, agent, config) {
        this.envs = lagEnvs;
        this.agent = agent;
        this.config = config;
        this.numEnvs = lagEnvs.length;
        this.stateDim = agent.stateDim;
        this.K = config.training.actionRepeat;
        this.gamma = config.sac.gamma;
        this.nStep = config.sac.nStep;
        this.mirror = config.nfsp.mirrorAugment;

        this.replay = new ReplayBuffer(config.nfsp.replayCapacity, this.stateDim);
        this.reservoir = new ReservoirBuffer(config.nfsp.reservoirCapacity, this.stateDim);

        this.buffer = new SnapshotBuffer(
            config.eval.snapshotBufferSize,
            () => buildAvgNet(this.stateDim, agent.numActions, config.network)
        );
        this.retired = []; // evicted snapshots possibly still mid-episode
        this.currentRating = config.eval.initialRating;

        this.episodeCount = 0;
        this.tickCount = 0;
        this.envSteps = 0;
        // Wall-clock breakdown per phase (ms), for the HUD profiler.
        this.perf = { decideMs: 0, physMs: 0, trainMs: 0, ticks: 0 };
        this.evalResults = [];  // recent eval scores (current AVG vs snapshots)
        this.brResults = [];    // recent BR-vs-AVG scores (exploitability proxy)

        this.slots = lagEnvs.map((env) => this.#freshSlot(env));
    }

    #freshSlot(env) {
        const slot = {
            env,
            pending: [null, null],      // in-progress decision transitions
            nq: [[], []],               // per-seat n-step queues of {s, a, r}
            modes: null,
            isEval: false,
            evalSeat: 0,
        };
        env.reset();
        this.#selectModes(slot);
        return slot;
    }

    #selectModes(slot) {
        const { evalProb } = this.config.eval;
        const eta = this.config.nfsp.anticipatory;
        if (this.buffer.size > 0 && Math.random() < evalProb) {
            slot.isEval = true;
            slot.evalSeat = Math.random() < 0.5 ? 0 : 1;
            const item = this.buffer.sample();
            slot.modes = [null, null];
            slot.modes[slot.evalSeat] = { kind: "avg" };
            slot.modes[1 - slot.evalSeat] = { kind: "snap", item };
        } else {
            slot.isEval = false;
            slot.modes = [0, 1].map(() => ({ kind: Math.random() < eta ? "br" : "avg" }));
        }
    }

    // --- n-step assembly ------------------------------------------------------
    // Feed one completed decision-level step; emit matured n-step transitions
    // into the replay buffer (plus mirrors).
    #pushStep(slot, seat, s, a, r, s2, done) {
        const q = slot.nq[seat];
        q.push({ s, a, r });
        if (done) {
            // Episode over: flush every queued entry with its discounted tail.
            for (let j = 0; j < q.length; j++) {
                let G = 0;
                for (let i = q.length - 1; i >= j; i--) G = q[i].r + this.gamma * G;
                this.#emit(q[j].s, q[j].a, G, s2, 1, this.gamma ** (q.length - j));
            }
            q.length = 0;
        } else if (q.length >= this.nStep) {
            let G = 0;
            for (let i = q.length - 1; i >= 0; i--) G = q[i].r + this.gamma * G;
            this.#emit(q[0].s, q[0].a, G, s2, 0, this.gamma ** q.length);
            q.shift();
        }
    }

    #emit(s, a, r, s2, done, gammaN) {
        this.replay.push(s, a, r, s2, done, gammaN);
        if (this.mirror) {
            this.replay.push(mirrorObs(s), mirrorActionIndex(a), r, mirrorObs(s2), done, gammaN);
        }
    }

    // --- one vectorized tick ----------------------------------------------------
    // Decisions (where due) -> env tick everywhere -> episode bookkeeping.
    // Returns episode-end infos for logging.
    async tick() {
        const t0 = performance.now();
        // GLOBAL decision alignment: every slot decides on the same tick
        // (tickCount % K == 0). This concentrates all policy forwards + GPU
        // readbacks into 1 tick in K instead of dribbling a small readback on
        // almost every tick (slots drift apart as episodes reset otherwise, and
        // the fixed readback latency, not batch size, is what costs time). A
        // freshly-reset slot idles (no keys) for < K ticks until the boundary.
        const deciding = []; // { slot, seat, state }
        if (this.tickCount % this.K === 0) {
            for (const slot of this.slots) {
                for (const seat of [0, 1]) {
                    const state = slot.env.decisionState(seat);
                    // The fresh decision state completes the previous decision block.
                    const p = slot.pending[seat];
                    if (p) this.#pushStep(slot, seat, p.state, p.action, p.reward, state, false);
                    deciding.push({ slot, seat, state });
                }
            }
        }

        if (deciding.length) {
            // Group requests by acting policy for batched forwards.
            const groups = new Map(); // key model ref or "br"/"avg" -> requests
            for (const req of deciding) {
                const mode = req.slot.modes[req.seat];
                const key = mode.kind === "snap" ? mode.item.actor : mode.kind;
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(req);
            }
            // All groups' forwards + readbacks run concurrently (one await total
            // per tick instead of one per policy identity).
            const results = await Promise.all(
                [...groups].map(async ([key, reqs]) => [reqs, await this.agent.actBatch(key, reqs.map((r) => r.state), false)])
            );
            for (const [reqs, actions] of results) {
                reqs.forEach((req, i) => {
                    const a = actions[i];
                    req.slot.pending[req.seat] = { state: req.state, action: a, reward: 0 };
                    req.slot.env.setDecision(req.seat, a);
                    // BR play is what the average policy imitates.
                    if (!req.slot.isEval && req.slot.modes[req.seat].kind === "br") {
                        this.reservoir.push(req.state, a);
                        if (this.mirror) this.reservoir.push(mirrorObs(req.state), mirrorActionIndex(a));
                    }
                });
            }
        }

        const t1 = performance.now();
        const infos = [];
        for (const slot of this.slots) {
            const res = slot.env.tick();
            this.envSteps++;
            for (const seat of [0, 1]) {
                if (slot.pending[seat]) slot.pending[seat].reward += res.rewards[seat];
            }
            // Optional replay recorder observes slot 0 (post-tick, pre-reset).
            if (this.recorder && slot === this.slots[0]) this.recorder.frame(slot.env, res);
            if (res.done) infos.push(this.#finishEpisode(slot, res));
        }
        this.tickCount++;
        this.#disposeRetired();
        this.perf.decideMs += t1 - t0;
        this.perf.physMs += performance.now() - t1;
        this.perf.ticks++;
        return infos;
    }

    #finishEpisode(slot, res) {
        // Terminal: flush both seats' pending transitions and n-step queues.
        for (const seat of [0, 1]) {
            const p = slot.pending[seat];
            if (p) this.#pushStep(slot, seat, p.state, p.action, p.reward, p.state, true);
            slot.pending[seat] = null;
            slot.nq[seat].length = 0; // safety; #pushStep(done) already cleared it
        }

        const scoreFor = (seat) => {
            const other = 1 - seat;
            if (res.timeout || (res.dead[0] && res.dead[1])) return 0.5;
            return res.dead[other] && !res.dead[seat] ? 1 : 0;
        };

        if (slot.isEval) {
            const score = scoreFor(slot.evalSeat);
            const item = slot.modes[1 - slot.evalSeat].item;
            const [rNew, sNew] = eloUpdate(this.currentRating, item.rating, score, this.config.eval.kFactor);
            this.currentRating = rNew;
            item.rating = sNew;
            this.evalResults.push(score);
            if (this.evalResults.length > 200) this.evalResults.shift();
        } else {
            // Exploitability proxy: how often does the BR beat the AVG policy?
            const kinds = [slot.modes[0].kind, slot.modes[1].kind];
            if (kinds.includes("br") && kinds.includes("avg")) {
                const brSeat = kinds[0] === "br" ? 0 : 1;
                this.brResults.push(scoreFor(brSeat));
                if (this.brResults.length > 300) this.brResults.shift();
            }
        }

        this.episodeCount++;
        if (this.episodeCount % this.config.eval.snapshotIntervalEpisodes === 0) {
            const evicted = this.buffer.addFromWeights(
                this.agent.avg.getWeights(), this.currentRating, this.episodeCount
            );
            if (evicted) this.retired.push(evicted);
        }

        slot.env.reset();
        this.#selectModes(slot);
        return { episode: this.episodeCount, rating: this.currentRating };
    }

    #disposeRetired() {
        if (!this.retired.length) return;
        const inUse = new Set();
        for (const slot of this.slots) {
            for (const m of slot.modes) if (m?.kind === "snap") inUse.add(m.item);
        }
        this.retired = this.retired.filter((item) => {
            if (inUse.has(item)) return true;
            item.actor.dispose();
            return false;
        });
    }

    // --- interleaved learning ------------------------------------------------------
    trainTick() {
        const nfsp = this.config.nfsp;
        if (this.replay.size < nfsp.learnStart) return;
        // Fire on an OFF-decision tick (decisions land on tickCount % K === 0;
        // tickCount was already incremented this tick). Keeps the ~100ms of JS
        // kernel-enqueue from sitting between a decision's state-build and its
        // GPU readback. Assumes trainEveryTicks is a multiple of actionRepeat.
        if (this.tickCount % nfsp.trainEveryTicks !== 2) return;
        const t0 = performance.now();
        for (let i = 0; i < nfsp.sacStepsPerTick; i++) {
            this.agent.sacStep(this.replay.sample(this.config.sac.batchSize));
        }
        if (this.reservoir.size >= this.config.avg.batchSize) {
            for (let i = 0; i < nfsp.slStepsPerTick; i++) {
                this.agent.slStep(this.reservoir.sample(this.config.avg.batchSize));
            }
        }
        this.perf.trainMs += performance.now() - t0;
    }

    // --- monitoring ------------------------------------------------------------------
    evalWinRate() {
        if (!this.evalResults.length) return 0;
        return this.evalResults.reduce((a, b) => a + b, 0) / this.evalResults.length;
    }

    // BR winrate over AVG. Hovering near 0.5 means the average policy is hard to
    // exploit even by its own dedicated best response — the NFSP convergence sign.
    brWinRate() {
        if (!this.brResults.length) return 0.5;
        return this.brResults.reduce((a, b) => a + b, 0) / this.brResults.length;
    }

    // --- save / load --------------------------------------------------------------------
    // Buffers are intentionally NOT saved (hundreds of MB); a resumed run refills
    // them before learning resumes (learnStart gate applies again).
    async serialize() {
        return {
            version: 1,
            algo: "nfsp-sac",
            savedAt: new Date().toISOString(),
            stateDim: this.stateDim,
            progress: {
                episodeCount: this.episodeCount,
                envSteps: this.envSteps,
                currentRating: this.currentRating,
            },
            agent: await this.agent.serialize(),
            snapshots: await this.buffer.serialize(),
        };
    }

    loadState(obj) {
        if (obj.algo !== "nfsp-sac") throw new Error(`not an rl2 save (algo=${obj.algo})`);
        if (obj.stateDim !== this.stateDim) {
            throw new Error(`save stateDim ${obj.stateDim} != current ${this.stateDim}`);
        }
        this.agent.loadState(obj.agent);
        for (const item of this.retired) item.actor.dispose();
        this.retired = [];
        this.buffer.load(obj.snapshots || []);
        this.episodeCount = obj.progress?.episodeCount ?? 0;
        this.envSteps = obj.progress?.envSteps ?? 0;
        this.currentRating = obj.progress?.currentRating ?? this.config.eval.initialRating;
        for (const slot of this.slots) {
            slot.env.reset();
            slot.pending = [null, null];
            slot.nq = [[], []];
            this.#selectModes(slot);
        }
    }
}
