"""Numpy inference for rl2 checkpoint networks (no TF needed in Python).

networks.mjs builds: [Dense(no bias) -> LayerNorm -> relu] x hidden -> Dense.
model.getWeights() order is therefore:
  kernel0, ln0_gamma, ln0_beta, kernel1, ln1_gamma, ln1_beta, ..., out_kernel, out_bias
tf.js layerNormalization uses epsilon 1e-3 (Keras default).
"""

import json

import numpy as np

LN_EPS = 1e-3


class MLPPolicy:
    def __init__(self, weight_records):
        self.tensors = [np.array(r["data"], dtype=np.float32).reshape(r["shape"])
                        for r in weight_records]

    @classmethod
    def from_checkpoint(cls, path: str, net: str = "avg"):
        """net: "avg" (deploy policy) or "policy" (SAC best response)."""
        with open(path) as f:
            save = json.load(f)
        agent = save.get("agent", save)
        if net not in agent:
            raise KeyError(f"checkpoint has no '{net}' network (keys: {list(agent)})")
        return cls(agent[net])

    def logits(self, x: np.ndarray) -> np.ndarray:
        t = self.tensors
        h = np.asarray(x, dtype=np.float32)
        i = 0
        while i + 2 < len(t) - 2:  # hidden blocks: kernel, gamma, beta
            h = h @ t[i]
            mean = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)
            h = (h - mean) / np.sqrt(var + LN_EPS) * t[i + 1] + t[i + 2]
            h = np.maximum(h, 0.0)  # relu
            i += 3
        return h @ t[-2] + t[-1]

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        z = self.logits(obs.reshape(1, -1))[0]
        if greedy:
            return int(np.argmax(z))
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(np.random.choice(len(p), p=p))
