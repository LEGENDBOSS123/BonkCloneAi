"""ppo6 network: a plain tanh MLP, no normalization — the canonical PPO trunk.

Deliberately isolated from bonk.networks (which is LayerNorm+ReLU and matches the
old tfjs deploy / old checkpoints). Here every hidden layer is

    Linear(bias=True) -> Tanh

so absolute-position magnitude survives to every layer (LayerNorm normalized it
away, which is exactly what you do NOT want when the policy must infer the map
geometry from its own x,y). Glorot init suits tanh; the output starts near-zero
so the initial policy is ~uniform / the initial value ~0.

Records format (tfjs dense-layer order): [k0,b0, k1,b1, ..., k_out,b_out], each
kernel stored [in,out] (torch Linear.weight is [out,in], so it transposes on the
way in/out). `dims_from_records` infers the architecture from the 2-D kernels, so
eval/deploy rebuild at any hidden size without being told the shape.
"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim = in_dim, list(hidden), out_dim
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=True), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim, bias=True)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # glorot: matched to tanh
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out.weight, 0.0, 0.01)  # near-zero output at start
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.trunk(x))

    # --- tfjs-format (de)serialization: [k0,b0, k1,b1, ..., k_out,b_out] ------
    def to_records(self):
        recs = []
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                k = m.weight.detach().cpu().numpy().T  # [in, out]
                b = m.bias.detach().cpu().numpy()
                recs.append({"shape": list(k.shape), "data": k.flatten().tolist()})
                recs.append({"shape": list(b.shape), "data": b.tolist()})
        k = self.out.weight.detach().cpu().numpy().T
        b = self.out.bias.detach().cpu().numpy()
        recs.append({"shape": list(k.shape), "data": k.flatten().tolist()})
        recs.append({"shape": list(b.shape), "data": b.tolist()})
        return recs

    def load_records(self, recs):
        arrays = [np.array(r["data"], dtype=np.float32).reshape(r["shape"]) for r in recs]
        i = 0
        with torch.no_grad():
            for m in self.trunk:
                if isinstance(m, nn.Linear):
                    m.weight.copy_(torch.from_numpy(arrays[i].T.copy()))
                    m.bias.copy_(torch.from_numpy(arrays[i + 1]))
                    i += 2
            self.out.weight.copy_(torch.from_numpy(arrays[i].T.copy()))
            self.out.bias.copy_(torch.from_numpy(arrays[i + 1]))

    @staticmethod
    def dims_from_records(recs):
        """Infer (in_dim, hidden, out_dim) from the 2-D kernel records."""
        kernels = [r["shape"] for r in recs if len(r["shape"]) == 2]
        in_dim = kernels[0][0]
        hidden = [k[1] for k in kernels[:-1]]
        out_dim = kernels[-1][1]
        return in_dim, hidden, out_dim

    @classmethod
    def from_records(cls, recs):
        net = cls(*cls.dims_from_records(recs))
        net.load_records(recs)
        return net
