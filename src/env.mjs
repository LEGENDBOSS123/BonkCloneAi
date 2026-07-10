import { loadBonkMap, parseBonkMap } from "./bonkmap/bonkmap_parser.mjs";

// Headless 2-player RL environment wrapping a Map. No rendering — it drives both
// players from explicit actions, steps the fixed-timestep simulation, and reports
// per-player rewards. A player "dies" if it falls past the kill line (world y >
// deathY) or leaves the kill circle (distance from origin > deathR); when one
// player dies the other scores +1 and the episode ends.
//
// Per-player observation (10 numbers), same order for each player:
//   [ x, y, vx, vy, heavyValue, up, down, left, right, heavy ]
// where heavyValue = (heavy held ? heavyPower : 0) and the last five are the keys
// held last step (0/1).
//
// state(i) is a flat egocentric vector: the querying player's own 10 numbers,
// then for each opponent its 10 numbers followed by 4 relative numbers
// (opponent - self): [ dx, dy, dvx, dvy ]. For 2 players that's 10 + 10 + 4 = 24.
//
// Actions per player are booleans: { up, down, left, right, heavy } (a 5-element
// array [up,down,left,right,heavy] also works). step() and the reward/observation
// dicts are keyed { p1, p2 }.

// Kill thresholds in bonk pixels (matching the debug overlay); scaled to world
// meters by the map's ppm.
const KILL_LINE_Y = 250;
const KILL_RADIUS = 850;

// State layout, used by the horizontal-mirror augmentation. Per-player feature
// block is [x, y, vx, vy, heavyValue, up, down, left, right, heavy]; each
// opponent block is followed by a 4-number relative block [dx, dy, dvx, dvy].
const FEATURES_PER_PLAYER = 10;
const RELATIVE_FEATURES = 4;
// Indices within a feature block flipped by a horizontal (x -> -x) mirror.
const NEGATE_IN_BLOCK = [0, 2]; // x, vx
const SWAP_IN_BLOCK = [7, 8]; // left <-> right keys

// getInput() reads codes off a rawinput-like source; map an action onto those.
function toRawInput(action) {
    const held = {
        ArrowUp: action.up,
        ArrowDown: action.down,
        ArrowLeft: action.left,
        ArrowRight: action.right,
        KeyX: action.heavy,
    };
    return { isDown: (code) => !!held[code] };
}

function normalizeAction(a) {
    if (!a) return { up: false, down: false, left: false, right: false, heavy: false };
    if (Array.isArray(a)) {
        return { up: !!a[0], down: !!a[1], left: !!a[2], right: !!a[3], heavy: !!a[4] };
    }
    return { up: !!a.up, down: !!a.down, left: !!a.left, right: !!a.right, heavy: !!a.heavy };
}

export class Env {
    // `shaping` is optional dense reward on top of the sparse +1 win. Defaults are
    // 0, so out of the box the reward is exactly: +1 to the survivor when the
    // opponent dies, 0 otherwise. Enable via config to help bootstrap learning.
    //   survivalBonus  : added to each still-alive player every step.
    //   boundaryPenalty: subtracted from a player scaled by how close it is to the
    //                    kill boundary (0 at the origin, up to boundaryPenalty at
    //                    the line/circle).
    constructor(map, { shaping = {} } = {}) {
        this.map = map;
        this.shaping = { survivalBonus: 0, boundaryPenalty: 0, ...shaping };
        // Both players are agent-driven in RL (parser marks p2 non-controllable
        // for the interactive game); enable control for both here.
        for (const p of map.players) p.controllable = true;

        this.numPlayers = map.players.length;
        this.initialHeavyPower = map.players.map((p) => p.heavyPower);
        // Derive the observation length from an actual build (own features +
        // per-opponent features + relative block), so it stays correct if the
        // feature set changes.
        this.stateSize = this.state(0).length;

        const ppm = map.ppm || 1;
        this.deathY = KILL_LINE_Y / ppm;
        this.deathR2 = (KILL_RADIUS / ppm) ** 2;

        this.done = false;
    }

    // Load/parse a map and wrap it. Pass { map } to reuse an existing Map, { data }
    // to parse an already-loaded bonk JSON, or { mapUrl } (defaults to map1.json).
    static async create({ map, data, mapUrl, gravity = 20, shaping } = {}) {
        if (!map) {
            if (data) {
                map = await parseBonkMap(data, { gravity });
            } else {
                const url = mapUrl ?? new URL("./bonkmap/map1.json", import.meta.url);
                map = await loadBonkMap(url, { gravity });
            }
        }
        return new Env(map, { shaping });
    }

    // The 10-number feature vector for one player.
    #features(player) {
        const t = player.body.translation();
        const v = player.body.linvel();
        const input = player.currentInput || {};
        const heavyValue = input.heavy ? player.heavyPower : 0;
        return [
            t.x, t.y,
            v.x, v.y,
            heavyValue,
            input.up ? 1 : 0,
            input.down ? 1 : 0,
            input.left ? 1 : 0,
            input.right ? 1 : 0,
            input.heavy ? 1 : 0,
        ];
    }

    // Flat observation from player `index`'s perspective: self first, then, for
    // each opponent, its features followed by relative (opponent - self) position
    // and velocity.
    state(index) {
        const players = this.map.players;
        const self = players[index];
        const selfPos = self.body.translation();
        const selfVel = self.body.linvel();
        const out = this.#features(self);
        for (let i = 0; i < players.length; i++) {
            if (i === index) continue;
            const other = players[i];
            out.push(...this.#features(other));
            const oPos = other.body.translation();
            const oVel = other.body.linvel();
            out.push(oPos.x - selfPos.x, oPos.y - selfPos.y, oVel.x - selfVel.x, oVel.y - selfVel.y);
        }
        return out;
    }

    // Observations for both players, keyed { p1, p2 }.
    observations() {
        return { p1: this.state(0), p2: this.state(1) };
    }

    // --- Horizontal-symmetry helpers ---------------------------------------
    // The arena is symmetric under a left/right flip (x -> -x). These map a
    // state/action to its mirror image, which is an equally valid transition and
    // lets a rollout be duplicated for ~2x sample efficiency.

    // Mirror one flat state vector (a new array; the input is not modified).
    mirrorState(state) {
        const out = state.slice();
        const mirrorBlock = (off) => {
            for (const k of NEGATE_IN_BLOCK) out[off + k] = -out[off + k];
            const [a, b] = SWAP_IN_BLOCK;
            [out[off + a], out[off + b]] = [out[off + b], out[off + a]];
        };
        mirrorBlock(0); // self block
        let off = FEATURES_PER_PLAYER;
        for (let i = 0; i < this.numPlayers - 1; i++) {
            mirrorBlock(off); // opponent block
            off += FEATURES_PER_PLAYER;
            out[off] = -out[off]; // relative dx
            out[off + 2] = -out[off + 2]; // relative dvx
            off += RELATIVE_FEATURES;
        }
        return out;
    }

    // Mirror an action [up, down, left, right, heavy] -> swap left/right.
    mirrorAction(a) {
        return [a[0], a[1], a[3], a[2], a[4]];
    }

    #isDead(player) {
        const t = player.body.translation();
        if (t.y > this.deathY) return true; // fell past the kill line (+y is down)
        if (t.x * t.x + t.y * t.y > this.deathR2) return true; // left the kill circle
        return false;
    }

    // Optional dense shaping for a still-alive player (0 with default config).
    #shapingReward(player) {
        const { survivalBonus, boundaryPenalty } = this.shaping;
        if (!survivalBonus && !boundaryPenalty) return 0;
        let r = survivalBonus;
        if (boundaryPenalty) {
            const t = player.body.translation();
            // Danger in [0, 1]: how far toward either kill boundary the player is.
            const lineDanger = Math.max(0, t.y) / this.deathY;
            const circleDanger = Math.sqrt(t.x * t.x + t.y * t.y) / Math.sqrt(this.deathR2);
            const danger = Math.min(1, Math.max(lineDanger, circleDanger));
            r -= boundaryPenalty * danger;
        }
        return r;
    }

    // Normalize the { p1, p2 } / [a0, a1] action bundle into an ordered array.
    #normalizeInputs(inputs) {
        const arr = Array.isArray(inputs) ? inputs : [inputs?.p1, inputs?.p2];
        return this.map.players.map((_, i) => normalizeAction(arr[i]));
    }

    // Advance one tick. `inputs` holds an action per player ({ p1, p2 } or array).
    // Returns { state: { p1, p2 }, reward: { p1, p2 }, done }.
    step(inputs) {
        if (this.done) {
            // Episode already over; caller should reset(). Report no reward.
            return { state: this.observations(), reward: { p1: 0, p2: 0 }, done: true, dead: { p1: true, p2: true } };
        }

        const actions = this.#normalizeInputs(inputs);
        // Apply every player's action (forces/jump/mass), then step the world once
        // — same ordering Map.step uses, but with a distinct action per player.
        this.map.players.forEach((player, i) => player.control(toRawInput(actions[i])));
        this.map.physics.step();
        this.map.tick++;

        const dead = this.map.players.map((p) => this.#isDead(p));
        // A player scores +1 iff its opponent died and it survived; simultaneous
        // deaths are a draw (0/0).
        const reward = {
            p1: dead[1] && !dead[0] ? 1 : 0,
            p2: dead[0] && !dead[1] ? 1 : 0,
        };
        // Optional dense shaping for players still alive (no-op with default config).
        if (!dead[0]) reward.p1 += this.#shapingReward(this.map.players[0]);
        if (!dead[1]) reward.p2 += this.#shapingReward(this.map.players[1]);
        this.done = dead.some(Boolean);

        return { state: this.observations(), reward, done: this.done, dead: { p1: dead[0], p2: dead[1] } };
    }

    // Randomly assign spawn points to players (a fresh permutation each call), so
    // neither player is tied to a fixed spawn. Falls back to each player's own
    // spawn if the map didn't provide a spawn-point list.
    #assignSpawns() {
        const spawns = this.map.spawnPoints?.length
            ? this.map.spawnPoints
            : this.map.players.map((p) => ({ x: p.spawnx, y: p.spawny }));
        const order = spawns.map((_, i) => i);
        for (let i = order.length - 1; i > 0; i--) {
            const j = (Math.random() * (i + 1)) | 0;
            [order[i], order[j]] = [order[j], order[i]];
        }
        this.map.players.forEach((player, i) => {
            const sp = spawns[order[i % spawns.length]];
            player.body.setTranslation({ x: sp.x, y: sp.y });
            player.spawnx = sp.x;
            player.spawny = sp.y;
        });
    }

    // Reset to the start of an episode: players sent to randomly-assigned spawns,
    // still, base mass, cleared inputs. Returns the initial observations { p1, p2 }.
    reset() {
        this.#assignSpawns();
        this.map.players.forEach((player, i) => {
            player.body.setLinvel({ x: 0, y: 0 });
            player.body.setAdditionalMass(0);
            player.heavyPower = this.initialHeavyPower[i];
            player.currentInput = null;
        });
        this.map.tick = 0;
        this.done = false;
        return this.observations();
    }
}
