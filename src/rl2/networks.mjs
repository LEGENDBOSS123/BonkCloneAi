import * as tf from "@tensorflow/tfjs";

// MLP builders for rl2. Three kinds of head, all the same trunk shape:
//   Q net       : state -> numActions Q-values
//   BR policy   : state -> numActions logits (softmax over joint actions)
//   AVG policy  : state -> numActions logits (softmax; supervised classifier)
//
// Layer order per hidden block: Dense -> [LayerNorm] -> activation. LayerNorm
// carries its own bias/gain, so the preceding Dense drops its bias.

let uid = 0; // unique model names (tfjs requires distinct layer names per model)

function mlp(inputDim, hidden, outputDim, { useLayerNorm, activation, outputScale, name }) {
    const tag = `${name}_${uid++}`;
    const input = tf.input({ shape: [inputDim] });
    let x = input;
    hidden.forEach((units, i) => {
        x = tf.layers.dense({ units, useBias: !useLayerNorm, name: `${tag}_dense${i}` }).apply(x);
        if (useLayerNorm) x = tf.layers.layerNormalization({ name: `${tag}_ln${i}` }).apply(x);
        x = tf.layers.activation({ activation, name: `${tag}_act${i}` }).apply(x);
    });
    const output = tf.layers
        .dense({
            units: outputDim,
            name: `${tag}_out`,
            kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: outputScale }),
        })
        .apply(x);
    return tf.model({ inputs: input, outputs: output, name: tag });
}

export function buildQNet(stateDim, numActions, netCfg) {
    return mlp(stateDim, netCfg.hidden, numActions, {
        useLayerNorm: netCfg.useLayerNorm,
        activation: netCfg.activation,
        outputScale: netCfg.outputScale,
        name: "q",
    });
}

export function buildPolicyNet(stateDim, numActions, netCfg) {
    return mlp(stateDim, netCfg.hidden, numActions, {
        useLayerNorm: netCfg.useLayerNorm,
        activation: netCfg.activation,
        outputScale: netCfg.outputScale,
        name: "pol",
    });
}

export function buildAvgNet(stateDim, numActions, netCfg) {
    return mlp(stateDim, netCfg.hidden, numActions, {
        useLayerNorm: netCfg.useLayerNorm,
        activation: netCfg.activation,
        outputScale: netCfg.outputScale,
        name: "avg",
    });
}

// --- weight (de)serialization (same {shape, data} records as rl1 saves) ------
// The clones are taken SYNCHRONOUSLY before any await: getWeights() returns the
// live variable tensors, and training keeps running between our async readbacks
// — an optimizer .assign() would dispose the live data mid-read (moveData /
// "reading 'backend'" crash). Clones are ours and immune to that.
export async function weightsToJSON(model) {
    const clones = model.getWeights().map((w) => w.clone());
    try {
        return await Promise.all(
            clones.map(async (c) => ({ shape: c.shape, data: Array.from(await c.data()) }))
        );
    } finally {
        clones.forEach((c) => c.dispose());
    }
}

export function weightsFromJSON(model, records) {
    const tensors = records.map((r) => tf.tensor(r.data, r.shape));
    model.setWeights(tensors);
    tensors.forEach((t) => t.dispose());
}
