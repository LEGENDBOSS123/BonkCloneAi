import * as tf from "@tensorflow/tfjs";

// Builds the actor and critic as two *separate* MLPs (no shared backbone), each
// with optional LayerNormalization after every hidden Dense layer.
//
// Layer order per hidden block:  Dense -> [LayerNorm] -> activation
//
//   Actor  : state -> hidden* -> Dense(actionDim)  (raw logits; Bernoulli per dim)
//   Critic : state -> hidden* -> Dense(1)           (state value)

function mlp(inputDim, hidden, outputDim, { useLayerNorm, activation, outputKernelScale = 1, name }) {
    const input = tf.input({ shape: [inputDim] });
    let x = input;
    hidden.forEach((units, i) => {
        x = tf.layers.dense({ units, useBias: !useLayerNorm, name: `${name}_dense${i}` }).apply(x);
        // LayerNorm has its own bias/gain, so the preceding Dense drops its bias.
        if (useLayerNorm) x = tf.layers.layerNormalization({ name: `${name}_ln${i}` }).apply(x);
        x = tf.layers.activation({ activation, name: `${name}_act${i}` }).apply(x);
    });
    const output = tf.layers
        .dense({
            units: outputDim,
            name: `${name}_out`,
            kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 0.01 * outputKernelScale }),
        })
        .apply(x);
    return tf.model({ inputs: input, outputs: output, name });
}

// Actor: outputs `actionDim` logits, one per independent Bernoulli action.
export function buildActor(stateDim, actionDim, netCfg) {
    return mlp(stateDim, netCfg.actorHidden, actionDim, {
        useLayerNorm: netCfg.useLayerNorm,
        activation: netCfg.activation,
        outputKernelScale: netCfg.policyLogitScale ?? 0.01,
        name: "actor",
    });
}

// Critic: outputs a single scalar state value.
export function buildCritic(stateDim, netCfg) {
    return mlp(stateDim, netCfg.criticHidden, 1, {
        useLayerNorm: netCfg.useLayerNorm,
        activation: netCfg.activation,
        outputKernelScale: 1,
        name: "critic",
    });
}
