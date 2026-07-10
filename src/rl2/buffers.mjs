// Preallocated typed-array buffers for NFSP.
//
// ReplayBuffer  : circular FIFO of decision-level n-step transitions for the SAC
//                 best response. Fixed Float32/Int32 storage — no per-push
//                 allocation, so multi-hour runs don't churn the GC.
// ReservoirBuffer: uniform-over-history sample of (state, action) pairs for the
//                 average-policy classifier (classic reservoir sampling, so the
//                 kept set is an unbiased sample of everything ever offered).

export class ReplayBuffer {
    constructor(capacity, stateDim) {
        this.capacity = capacity;
        this.stateDim = stateDim;
        this.states = new Float32Array(capacity * stateDim);
        this.nextStates = new Float32Array(capacity * stateDim);
        this.actions = new Int32Array(capacity);
        this.rewards = new Float32Array(capacity);
        this.dones = new Float32Array(capacity);   // 1 if s' is terminal
        this.gammaN = new Float32Array(capacity);  // gamma^n for this n-step hop
        this.size = 0;
        this.head = 0;
    }

    push(state, action, reward, nextState, done, gammaN) {
        const i = this.head;
        this.states.set(state, i * this.stateDim);
        this.nextStates.set(nextState, i * this.stateDim);
        this.actions[i] = action;
        this.rewards[i] = reward;
        this.dones[i] = done ? 1 : 0;
        this.gammaN[i] = gammaN;
        this.head = (i + 1) % this.capacity;
        if (this.size < this.capacity) this.size++;
    }

    // Uniform sample of n transitions into fresh typed arrays (batch-shaped).
    sample(n) {
        const D = this.stateDim;
        const states = new Float32Array(n * D);
        const nextStates = new Float32Array(n * D);
        const actions = new Int32Array(n);
        const rewards = new Float32Array(n);
        const dones = new Float32Array(n);
        const gammaN = new Float32Array(n);
        for (let k = 0; k < n; k++) {
            const i = (Math.random() * this.size) | 0;
            states.set(this.states.subarray(i * D, i * D + D), k * D);
            nextStates.set(this.nextStates.subarray(i * D, i * D + D), k * D);
            actions[k] = this.actions[i];
            rewards[k] = this.rewards[i];
            dones[k] = this.dones[i];
            gammaN[k] = this.gammaN[i];
        }
        return { states, actions, rewards, nextStates, dones, gammaN, n };
    }
}

export class ReservoirBuffer {
    constructor(capacity, stateDim) {
        this.capacity = capacity;
        this.stateDim = stateDim;
        this.states = new Float32Array(capacity * stateDim);
        this.actions = new Int32Array(capacity);
        this.size = 0;
        this.seen = 0; // total items ever offered (drives the reservoir odds)
    }

    push(state, action) {
        this.seen++;
        if (this.size < this.capacity) {
            this.states.set(state, this.size * this.stateDim);
            this.actions[this.size] = action;
            this.size++;
            return;
        }
        // Replace a random slot with probability capacity/seen.
        const j = (Math.random() * this.seen) | 0;
        if (j < this.capacity) {
            this.states.set(state, j * this.stateDim);
            this.actions[j] = action;
        }
    }

    sample(n) {
        const D = this.stateDim;
        const states = new Float32Array(n * D);
        const actions = new Int32Array(n);
        for (let k = 0; k < n; k++) {
            const i = (Math.random() * this.size) | 0;
            states.set(this.states.subarray(i * D, i * D + D), k * D);
            actions[k] = this.actions[i];
        }
        return { states, actions, n };
    }
}
