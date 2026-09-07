// Pure-JS inference for bonk agents. No TensorFlow, no CDN, no WebGL.
//
// WHY: the deployed actor is ~79k MACs (1.2M MAC/s at 15 decisions/s). Pulling
// 1 MB of tfjs from jsDelivr to run that is absurd, and it cost us real bugs:
// a WebGL backend that contended with the game's renderer (4 ms vs 0.58 ms),
// an async readback that added ~2 frames of input lag, and — for the recurrent
// model — Keras's GRU gate order (z,r,h) silently disagreeing with PyTorch's
// (r,z,n). Doing the arithmetic ourselves removes that entire class of problem
// and makes the paste-in script self-contained: no network dependency at all.
//
// THIS FILE IS THE TESTED SOURCE OF TRUTH. play3.mjs inlines a copy (a paste-in
// script cannot import); tools/test_net_parity.mjs checks both the numerics
// against PyTorch and that the inlined copy has not drifted.

// ── primitives ───────────────────────────────────────────────────────────────

// Kernel in tfjs record layout: flat [in, out], row-major. y = xW.
export function matvecTfjs(kernel, inDim, outDim, x, bias) {
    const y = new Float32Array(outDim);
    if (bias) y.set(bias);
    for (let i = 0; i < inDim; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const row = i * outDim;
        for (let j = 0; j < outDim; j++) y[j] += xi * kernel[row + j];
    }
    return y;
}

// Torch Linear layout: nested [out][in]. y = Wx + b.
export function matvecTorch(W, x, bias) {
    const outDim = W.length;
    const y = new Float32Array(outDim);
    for (let j = 0; j < outDim; j++) {
        const row = W[j];
        let s = bias ? bias[j] : 0;
        for (let i = 0; i < row.length; i++) s += row[i] * x[i];
        y[j] = s;
    }
    return y;
}

// Matches torch.nn.LayerNorm: biased variance (divide by N, not N-1).
export function layerNorm(x, gamma, beta, eps = 1e-3) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let varc = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mean; varc += d * d; }
    varc /= n;
    const inv = 1 / Math.sqrt(varc + eps);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    return y;
}

export function relu(x) {
    const y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = x[i] > 0 ? x[i] : 0;
    return y;
}

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// ── feedforward actor (tfjs record format) ───────────────────────────────────
// Records: k0, ln0_g, ln0_b, k1, ln1_g, ln1_b, ..., k_out, b_out.
// Layer block = [Linear(no bias) -> LayerNorm(eps 1e-3) -> ReLU], then a final
// Linear WITH bias. Same structure as bonk/networks.py MLP.
export function buildMLP(records) {
    const layers = [];
    let i = 0;
    while (i + 3 < records.length) {           // a trailing kernel+bias remain
        layers.push({
            k: Float32Array.from(records[i].data),
            inDim: records[i].shape[0], outDim: records[i].shape[1],
            g: Float32Array.from(records[i + 1].data),
            b: Float32Array.from(records[i + 2].data),
        });
        i += 3;
    }
    const outK = records[i], outB = records[i + 1];
    return {
        layers,
        outK: Float32Array.from(outK.data),
        outIn: outK.shape[0], outOut: outK.shape[1],
        outB: Float32Array.from(outB.data),
        stateDim: layers.length ? layers[0].inDim : outK.shape[0],
        actionDim: outK.shape[1],
    };
}

export function mlpForward(net, obs) {
    let x = Float32Array.from(obs);
    for (const L of net.layers) {
        x = relu(layerNorm(matvecTfjs(L.k, L.inDim, L.outDim, x, null), L.g, L.b));
    }
    return matvecTfjs(net.outK, net.outIn, net.outOut, x, net.outB);
}

// ── recurrent actor (RecurrentAC.to_records) ─────────────────────────────────
export function buildGRU(obj) {
    const t = obj.tensors;
    const enc = [];
    // enc is Sequential(Linear(no bias), LayerNorm, ReLU) repeated.
    for (let li = 0; ; li += 3) {
        if (!(`enc.${li}.weight` in t)) break;
        enc.push({
            W: t[`enc.${li}.weight`],
            g: t[`enc.${li + 1}.weight`],
            b: t[`enc.${li + 1}.bias`],
        });
    }
    return {
        enc,
        Wih: t["cell.weight_ih"], Whh: t["cell.weight_hh"],
        bih: t["cell.bias_ih"], bhh: t["cell.bias_hh"],
        piW: t["pi.weight"], piB: t["pi.bias"],
        hidden: obj.hidden, stateDim: obj.stateDim, actionDim: obj.numActions,
    };
}

// EXACT torch.nn.GRUCell semantics. Gate order in weight_ih/weight_hh is
// [r, z, n]; crucially the reset gate multiplies (W_hn h + b_hn) INCLUDING the
// hidden bias. Keras orders gates (z, r, h) and applies the reset differently —
// that mismatch is the classic silent-corruption bug this avoids.
export function gruStep(net, obs, h) {
    let x = Float32Array.from(obs);
    for (const L of net.enc) x = relu(layerNorm(matvecTorch(L.W, x, null), L.g, L.b));

    const H = net.hidden;
    const gi = matvecTorch(net.Wih, x, net.bih);   // [3H]
    const gh = matvecTorch(net.Whh, h, net.bhh);   // [3H]
    const hn = new Float32Array(H);
    for (let j = 0; j < H; j++) {
        const r = sigmoid(gi[j] + gh[j]);
        const z = sigmoid(gi[H + j] + gh[H + j]);
        const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
        hn[j] = (1 - z) * n + z * h[j];
    }
    return { logits: matvecTorch(net.piW, hn, net.piB), h: hn };
}

// ── unified entry ────────────────────────────────────────────────────────────
export function buildPolicy(save) {
    const agent = save.agent || save;
    if (agent.recurrent) {
        const net = buildGRU(agent.recurrent);
        return { kind: "gru", net, hidden: net.hidden,
                 stateDim: net.stateDim, actionDim: net.actionDim };
    }
    const net = buildMLP(agent.actor);
    return { kind: "mlp", net, hidden: 0,
             stateDim: net.stateDim, actionDim: net.actionDim };
}

export function policyStep(p, obs, h) {
    if (p.kind === "gru") return gruStep(p.net, obs, h);
    return { logits: mlpForward(p.net, obs), h };
}

export function argmax(a) {
    let b = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
    return b;
}

export function sampleSoftmax(a) {
    let m = -Infinity;
    for (const v of a) if (v > m) m = v;
    let sum = 0;
    const p = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) { p[i] = Math.exp(a[i] - m); sum += p[i]; }
    let r = Math.random() * sum;
    for (let i = 0; i < a.length; i++) { r -= p[i]; if (r <= 0) return i; }
    return a.length - 1;
}
