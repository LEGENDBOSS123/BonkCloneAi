import * as tf from "@tensorflow/tfjs";

// Standard ELO. Expected score of A against B.
export function expectedScore(ratingA, ratingB) {
    return 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
}

// New [ratingA, ratingB] after a match where A scored `scoreA` in {0, 0.5, 1}.
export function eloUpdate(ratingA, ratingB, scoreA, k) {
    const ea = expectedScore(ratingA, ratingB);
    const eb = 1 - ea;
    return [ratingA + k * (scoreA - ea), ratingB + k * (1 - scoreA - eb)];
}

// A fixed-capacity pool of frozen actor snapshots, each a materialized actor
// model (so opponents can be forwarded in batches by identity) plus its own ELO
// rating. `modelFactory()` builds a fresh, correctly-shaped actor model. Dropping
// a snapshot disposes its model.
export class SnapshotBuffer {
    constructor(maxSize, modelFactory) {
        this.maxSize = maxSize;
        this.modelFactory = modelFactory;
        this.items = []; // { actor: tf.LayersModel, rating: number, episode: number }
    }

    get size() {
        return this.items.length;
    }

    // Push a snapshot; if it overflows, return the EVICTED item (undisposed) so
    // the caller can dispose it once no one references it (a slot may still be
    // mid-episode against it). Returns null if nothing was evicted.
    #push(item) {
        this.items.push(item);
        return this.items.length > this.maxSize ? this.items.shift() : null;
    }

    // Snapshot from a set of weight tensors (e.g. the live actor's getWeights()).
    // setWeights copies the data, so the caller keeps ownership of `weights`.
    // Returns the evicted snapshot (or null) — caller owns its disposal.
    addFromWeights(weights, rating, episode) {
        const actor = this.modelFactory();
        actor.setWeights(weights);
        return this.#push({ actor, rating, episode });
    }

    sample() {
        if (!this.items.length) return null;
        return this.items[(Math.random() * this.items.length) | 0];
    }

    // Dispose every snapshot's model (used before loading a fresh state).
    clear() {
        for (const item of this.items) item.actor.dispose();
        this.items = [];
    }

    // Serializable form: ratings/episodes + weights as {shape, data} records.
    async serialize() {
        return Promise.all(
            this.items.map(async (it) => ({
                rating: it.rating,
                episode: it.episode,
                weights: await Promise.all(
                    it.actor.getWeights().map(async (w) => ({ shape: w.shape, data: Array.from(await w.data()) }))
                ),
            }))
        );
    }

    // Rebuild from serialized records (replaces current contents).
    load(records) {
        this.clear();
        for (const rec of records) {
            const actor = this.modelFactory();
            const tensors = rec.weights.map((w) => tf.tensor(w.data, w.shape));
            actor.setWeights(tensors);
            tensors.forEach((t) => t.dispose());
            this.items.push({ actor, rating: rec.rating, episode: rec.episode });
        }
    }
}
