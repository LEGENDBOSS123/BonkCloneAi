// Decision-level wrapper around the core Env for rl2.
//
// Adds, on top of src/env.mjs:
//   - JOINT discrete actions: 18 = 3 (none/left/right) x 3 (none/up/down) x 2
//     (heavy); index = lr*6 + ud*2 + heavy; index 0 = no keys.
//   - INPUT LAG: a decision set at tick t takes effect at tick t+inputLag; the
//     previously-applied action keeps holding until then (simulated reaction /
//     network delay, so the policy transfers to real bonk).
//   - NORMALIZED observations with the applied keys, the last *decided* action
//     (pending block — differs from applied keys under real lag), and an
//     episode-clock feature (fraction of maxEpisodeSteps elapsed, so the agent
//     sees the draw approaching).
//   - Terminal rewards from config (win/loss/draw); no dense shaping.
//
// Observation layout (D = 30):
//   [ 0- 9] self  : x, y, vx, vy, heavyValue, up, down, left, right, heavyKey
//   [10-19] opp   : same 10 features
//   [20-23] rel   : dx, dy, dvx, dvy   (opponent - self)
//   [24-28] pend  : own last DECIDED action bits [up, down, left, right, heavy]
//   [29]    time  : episodeSteps / maxEpisodeSteps in [0, 1]
// Key bits in the self/opp blocks are the actions currently APPLIED to physics.

const IDLE = 0; // action index with no keys held

export function indexToKeys(index) {
    const lr = (index / 6) | 0;
    const ud = ((index / 2) | 0) % 3;
    return {
        left: lr === 1,
        right: lr === 2,
        up: ud === 1,
        down: ud === 2,
        heavy: index % 2 === 1,
    };
}

export function keysToIndex(k) {
    const lr = k.left ? 1 : k.right ? 2 : 0;
    const ud = k.up ? 1 : k.down ? 2 : 0;
    return lr * 6 + ud * 2 + (k.heavy ? 1 : 0);
}

// Horizontal mirror of a joint action: swap left <-> right.
export function mirrorActionIndex(index) {
    const lr = (index / 6) | 0;
    const rest = index % 6;
    const mlr = lr === 1 ? 2 : lr === 2 ? 1 : 0;
    return mlr * 6 + rest;
}

const FEATS = 10; // per-player block size
const REL = 4;
const PEND = 5;
export const STATE_DIM = FEATS * 2 + REL + PEND + 1;

// Obs indices touched by a horizontal (x -> -x) mirror.
const MIRROR_NEGATE = [0, 2, FEATS, FEATS + 2, 20, 22]; // x, vx (both), dx, dvx
const MIRROR_SWAPS = [
    [7, 8],                       // self left/right (applied)
    [FEATS + 7, FEATS + 8],       // opp left/right (applied)
    [24 + 2, 24 + 3],             // pending left/right
];

export function mirrorObs(obs) {
    const out = obs.slice();
    for (const i of MIRROR_NEGATE) out[i] = -out[i];
    for (const [a, b] of MIRROR_SWAPS) {
        const t = out[a];
        out[a] = out[b];
        out[b] = t;
    }
    return out;
}

export class LagEnv {
    constructor(coreEnv, config) {
        this.core = coreEnv;
        this.cfg = config;
        this.lag = config.env.inputLag;
        this.maxSteps = config.env.maxEpisodeSteps;
        this.obsCfg = config.obs;
        this.stateDim = STATE_DIM;
        // Per-seat action state: previous & latest decision, ticks since latest.
        this.seats = [0, 1].map(() => ({ prev: IDLE, last: IDLE, since: 1e9 }));
        this.episodeSteps = 0;
    }

    reset() {
        this.core.reset();
        for (const s of this.seats) {
            s.prev = IDLE;
            s.last = IDLE;
            s.since = 1e9; // "long ago": the idle action is fully applied
        }
        this.episodeSteps = 0;
    }

    // Register seat i's new decision. Takes effect `lag` ticks from now; until
    // then the previous decision keeps applying.
    setDecision(i, actionIndex) {
        const s = this.seats[i];
        s.prev = s.last;
        s.last = actionIndex;
        s.since = 0;
    }

    // The action index currently governing seat i's physics.
    #appliedIndex(i) {
        const s = this.seats[i];
        return s.since < this.lag ? s.prev : s.last;
    }

    // Public view of both seats' currently-applied action indices (replays).
    appliedActions() {
        return [this.#appliedIndex(0), this.#appliedIndex(1)];
    }

    // Advance one physics tick using each seat's lag-effective action.
    // Returns { rewards: [r0, r1], done, dead: [d0, d1], timeout }.
    tick() {
        const k0 = indexToKeys(this.#appliedIndex(0));
        const k1 = indexToKeys(this.#appliedIndex(1));
        const res = this.core.step({ p1: k0, p2: k1 });
        this.seats[0].since++;
        this.seats[1].since++;
        this.episodeSteps++;

        const dead = [res.dead.p1, res.dead.p2];
        const timeout = !res.done && this.episodeSteps >= this.maxSteps;
        const done = res.done || timeout;

        // Terminal rewards from config; zero otherwise (sparse, no shaping).
        const { winReward, lossReward, drawReward } = this.cfg.env;
        let rewards = [0, 0];
        if (done) {
            if (timeout || (dead[0] && dead[1])) rewards = [drawReward, drawReward];
            else if (dead[1]) rewards = [winReward, lossReward];
            else rewards = [lossReward, winReward];
        }
        return { rewards, done, dead, timeout };
    }

    // Normalized observation for seat i (built at decision points).
    decisionState(i) {
        const { posScale, velScale, heavyScale } = this.obsCfg;
        const players = this.core.map.players;
        const self = players[i];
        const opp = players[1 - i];
        const out = new Float32Array(STATE_DIM);

        const writeBlock = (off, player, appliedIdx) => {
            const t = player.body.translation();
            const v = player.body.linvel();
            const k = indexToKeys(appliedIdx);
            out[off] = t.x * posScale;
            out[off + 1] = t.y * posScale;
            out[off + 2] = v.x * velScale;
            out[off + 3] = v.y * velScale;
            out[off + 4] = k.heavy ? player.heavyPower * heavyScale : 0;
            out[off + 5] = k.up ? 1 : 0;
            out[off + 6] = k.down ? 1 : 0;
            out[off + 7] = k.left ? 1 : 0;
            out[off + 8] = k.right ? 1 : 0;
            out[off + 9] = k.heavy ? 1 : 0;
        };
        writeBlock(0, self, this.#appliedIndex(i));
        writeBlock(FEATS, opp, this.#appliedIndex(1 - i));

        const ts = self.body.translation();
        const to = opp.body.translation();
        const vs = self.body.linvel();
        const vo = opp.body.linvel();
        out[20] = (to.x - ts.x) * posScale;
        out[21] = (to.y - ts.y) * posScale;
        out[22] = (vo.x - vs.x) * velScale;
        out[23] = (vo.y - vs.y) * velScale;

        // Pending block: the latest DECIDED action (may not be applied yet). In
        // deployment this is "the keys I just sent", which real lag separates
        // from the applied keys bonk reports.
        const pk = indexToKeys(this.seats[i].last);
        out[24] = pk.up ? 1 : 0;
        out[25] = pk.down ? 1 : 0;
        out[26] = pk.left ? 1 : 0;
        out[27] = pk.right ? 1 : 0;
        out[28] = pk.heavy ? 1 : 0;

        out[29] = Math.min(1, this.episodeSteps / this.maxSteps);
        return out;
    }
}
