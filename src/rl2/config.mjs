// Central config for the rl2 (NFSP + discrete SAC) setup. Everything tunable
// lives here so the env wrapper, agent, trainer, and browser entry read from one
// place.
//
// Algorithm summary (Neural Fictitious Self-Play):
//   - Best response (BR): discrete-action SAC (twin Q + softmax policy +
//     auto-tuned entropy temperature) trained off-policy from a replay buffer.
//   - Average policy (AVG): a classifier trained by supervised learning on the
//     BR's historical (state, action) pairs, stored in a reservoir buffer so the
//     dataset is uniform over training history. AVG approximates the time-average
//     of best responses, which in 2p zero-sum converges toward a Nash strategy —
//     this is the policy you deploy against humans (hard to exploit).
//   - Each seat, each episode, plays BR with prob `anticipatory` (eta), else AVG.

export const CONFIG = {
    // --- Environment / simulation ------------------------------------------
    env: {
        gravity: 20,
        // Hard cap on episode length (physics ticks). Hitting it is a draw.
        // Early policies stall to the cap constantly, so a long cap mostly buys
        // dead time; 1500 (~50s of game) turns episodes over ~1.7x faster.
        maxEpisodeSteps: 1500,
        // Terminal rewards (zero-sum-ish). Draw slightly negative so stalling out
        // the clock is worse than winning but better than losing — pushes toward
        // decisive play without any dense shaping. Set to 0 for the pure game.
        winReward: 1,
        lossReward: -1,
        drawReward: -0.5,
        // Input lag: a decision made at tick t takes effect at tick t+inputLag
        // (the previously-applied action keeps holding until then). Simulates
        // human/network reaction delay so the trained policy transfers to real
        // bonk. Must be <= actionRepeat for the decision-aligned Markov state.
        inputLag: 2,
    },

    // --- Observation normalization -----------------------------------------
    // Raw features are divided by these so everything the net sees is O(1).
    // Generic scale constants, not map-specific knowledge.
    obs: {
        posScale: 1 / 30,   // world meters -> ~[-2, 2] inside the arena
        velScale: 1 / 30,   // m/s -> ~[-1.5, 1.5] at typical disc speeds
        heavyScale: 1 / 1000, // heavyPower 0..1000 -> [0, 1]
    },

    // --- Action space --------------------------------------------------------
    // Joint discrete: 3 (none/left/right) x 3 (none/up/down) x 2 (heavy) = 18.
    // index = lr * 6 + ud * 2 + heavy  (index 0 = no keys held).
    numActions: 18,

    // --- Rendering (browser visualizer) ------------------------------------
    render: {
        worldHeight: 52,        // meters visible vertically
        defaultGameSpeed: 64,   // top.gameSpeed initial value
        ticksPerSecondBase: 30, // rendered-env real-time rate at gameSpeed 1
        // Max ticks per event-loop turn (each tick advances ALL envs + does the
        // interleaved gradient steps). Keeps the UI responsive.
        maxTicksPerChunk: 64,
    },

    // --- Networks -------------------------------------------------------------
    network: {
        // [128,128] is ample for a 30-dim obs / 18 actions, and this machine's
        // tfjs matmul throughput is the binding constraint: vs [256,256] each
        // gradient step is ~4x cheaper. NOTE: changing this invalidates saves.
        hidden: [128, 128],     // Q nets, BR policy, and AVG policy all use this
        useLayerNorm: true,
        activation: "relu",
        // Small std on output layers so initial Q/logits are near 0.
        outputScale: 0.01,
    },

    // --- Discrete SAC (best response) ----------------------------------------
    sac: {
        gamma: 0.997,           // per DECISION (one decision = actionRepeat ticks)
        nStep: 3,               // n-step returns over decisions (sparse-reward reach)
        qLr: 3e-4,
        policyLr: 3e-4,
        alphaLr: 3e-4,
        initAlpha: 0.2,         // initial entropy temperature
        // target entropy = ratio * ln(numActions); alpha auto-tunes toward it.
        targetEntropyRatio: 0.5,
        // Floor for alpha. With sparse rewards the Q-surface starts flat, so H
        // stays above target no matter how small alpha gets — the tuner then
        // decays alpha toward 0 and the entropy regularizer vanishes entirely.
        minAlpha: 0.01,
        // Target-net Polyak update, thinned: every `polyakEvery` SAC steps with a
        // proportionally bigger tau (1-(1-0.005)^4 ≈ 0.02 — same timescale, 1/4
        // the kernel dispatches; the Polyak loop is ~130 tiny GPU ops).
        tau: 0.02,
        polyakEvery: 4,
        batchSize: 1024,
        // 0 = no grad clipping. The clip is ~150 extra dispatches per step and at
        // norm 10 it almost never triggered; standard SAC ships without it.
        maxGradNorm: 0,
    },

    // --- Average policy (supervised) -----------------------------------------
    avg: {
        lr: 1e-4,
        batchSize: 1024,
    },

    // --- NFSP ------------------------------------------------------------------
    nfsp: {
        // Probability a seat plays its BEST RESPONSE this episode (eta). The rest
        // of the time it plays the average policy. BR seats generate SL data.
        anticipatory: 0.15,
        replayCapacity: 300_000,    // RL transitions (decision-level, n-step)
        reservoirCapacity: 500_000, // SL (state, action) pairs, reservoir-sampled
        learnStart: 5_000,          // RL transitions before any gradient steps
        // Gradient steps interleaved with collection: every `trainEveryTicks`
        // env ticks, run this many SAC and SL steps. Wall-clock is dominated by
        // GPU dispatch count, not batch size — so train RARELY with BIG batches
        // (2048 every 8 ticks ≈ same samples/sec as 512 every tick or two, at a
        // fraction of the overhead).
        trainEveryTicks: 8,
        sacStepsPerTick: 1,
        slStepsPerTick: 1,
        // Duplicate each stored sample with its horizontal mirror (valid on a
        // left/right-symmetric arena only).
        mirrorAugment: true,
    },

    // --- Evaluation / ELO ------------------------------------------------------
    eval: {
        // Fraction of episodes that are rated eval matches: current AVG policy vs
        // a frozen AVG snapshot. (Their transitions still feed the replay buffer —
        // off-policy learning welcomes any data — but never the SL buffer.)
        evalProb: 0.06,
        snapshotIntervalEpisodes: 500, // snapshot the AVG policy every N episodes
        snapshotBufferSize: 30,
        initialRating: 1000,
        kFactor: 16,
    },

    // --- Training loop -----------------------------------------------------------
    training: {
        totalEpisodes: 300_000,
        logIntervalEpisodes: 25,
        // Physics is cheap (2-body planck worlds); with gradient work now paid
        // per-8-ticks rather than per-env, more envs = nearly-free extra data.
        numEnvs: 48,
        // Decide every K ticks; the chosen action (after inputLag) holds for the
        // block. gamma above is per K-tick decision.
        actionRepeat: 4,
        // Refresh HUD loss/entropy stats every N gradient steps (async readback).
        statsEvery: 50,
    },
};

export default CONFIG;
