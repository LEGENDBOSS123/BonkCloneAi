// Central config for the RL setup. Everything tunable lives here so the agent,
// trainer, self-play/ELO, and browser entry read from one place.

export const CONFIG = {
    // --- Environment / simulation ------------------------------------------
    env: {
        gravity: 20,
        // Hard cap on episode length (ticks). Prevents endless stalemates; the
        // episode is treated as a draw if it hits this without a death.
        maxEpisodeSteps: 2500,
        // Dense reward shaping added on top of the sparse +1 win. Both 0 => pure
        // sparse reward (the chosen spec). Raise to help bootstrap learning.
        shaping: {
            survivalBonus: 0,   // per-step reward for each living player
            boundaryPenalty: 0, // per-step penalty scaled by nearness to the kill boundary
        },
    },

    // --- Rendering (browser visualizer) ------------------------------------
    render: {
        worldHeight: 52,        // meters visible vertically
        defaultGameSpeed: 64,    // top.gameSpeed initial value
        // Time-based pacing in "ticks" (one vectorized step of ALL envs). At
        // gameSpeed = 1 the rendered env runs at ticksPerSecondBase (~real-time);
        // total env throughput is that times numEnvs. Driven by a Web Worker
        // metronome so it keeps running in a background tab.
        ticksPerSecondBase: 30,
        // Max ticks per event-loop turn — keeps each chunk short so the renderer /
        // UI stays responsive (each tick already advances numEnvs envs).
        maxTicksPerChunk: 64,
    },

    // --- Network (separate actor & critic, no shared backbone) --------------
    network: {
        actorHidden: [256, 256],
        criticHidden: [256, 256],
        useLayerNorm: true,
        activation: "tanh", // "relu" or "tanh"
        // Small std for the final policy-logit layer so early actions are near
        // 50/50 (logits ~ 0) instead of saturated.
        policyLogitScale: 0.01,
    },

    // --- PPO ----------------------------------------------------------------
    ppo: {
        gamma: 0.997,
        gaeLambda: 0.95,
        clipEpsilon: 0.2,
        actorLr: 2e-4,
        criticLr: 3e-4,
        epochs: 3,              // optimization passes over each rollout
        // Update wall-clock is dominated by the NUMBER of minibatch iterations
        // (tfjs per-op dispatch overhead; the net is tiny) = effectiveBatch *
        // epochs / minibatchSize. The effective batch is ~4x rolloutSteps
        // (opponent-POV x2, mirror x2) -> with rolloutSteps=32768 that's ~131k
        // rows. Big minibatches are the main speed lever; lower for more gradient
        // steps per update.
        minibatchSize: 1024*8,
        rolloutSteps: 2048*8,     // learner transitions collected per PPO update
        entropyCoef: 0.008,      // exploration bonus on the actor
        valueCoef: 0.5,         // scales the critic loss
        maxGradNorm: 0.5,       // global grad-norm clip (per network)
        normalizeAdvantages: true,
        clipValueLoss: true,    // PPO-style clipped value loss
        // Duplicate each transition with its horizontal mirror (state+action) for
        // ~2x data. Only valid if the arena is left/right symmetric — turn off for
        // asymmetric maps.
        mirrorAugment: true,
    },

    // --- Self-play snapshot buffer -----------------------------------------
    selfPlay: {
        // Every N finished episodes, snapshot the current actor weights into the
        // buffer (the opponent pool).
        snapshotIntervalEpisodes: 2000,
        bufferSize: 40,         // max snapshots kept (oldest dropped)
        // Each episode: probability the opponent is the *current* (live) model
        // rather than a frozen snapshot sampled from the buffer.
        opponentCurrentProb: 0.5,
        // Sample opponent actions stochastically (true) or greedily (false).
        opponentStochastic: true,
    },

    // --- ELO ----------------------------------------------------------------
    elo: {
        initialRating: 1000,
        kFactor: 16,
        // ELO only updates on episodes where the opponent is a *snapshot* (a rated
        // entity); mirror matches vs the current model are skipped.
    },

    // --- Training loop ------------------------------------------------------
    training: {
        totalEpisodes: 150000,  // stop after this many finished episodes
        logIntervalEpisodes: 10,
        // Which player index the learner controls (the other is the opponent).
        learnerIndex: 0,
        // Number of parallel (vectorized) envs. Each tick advances all of them
        // with one batched forward/readback — the main throughput lever. Raise
        // until the main thread's physics (CPU) becomes the bottleneck.
        numEnvs: 24,
        // Action repeat (frame-skip): the policy picks an action once, then holds
        // it for this many frames (rewards accumulate into one transition). Cuts
        // policy forwards ~this many times and lets the agent commit to actions.
        // 1 = decide every frame. Note: with K>1 the discount `gamma` is per
        // decision (i.e. per K frames), so raising K lengthens the effective
        // horizon — retune gamma if needed.
        actionRepeat: 4,
    },
};

export default CONFIG;
