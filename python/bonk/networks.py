"""Torch networks matching src/rl2/networks.mjs exactly, with lossless
conversion to/from the tfjs save format so checkpoints work in the browser.

Architecture per net: [Linear(no bias) -> LayerNorm(eps=1e-3) -> ReLU] x hidden
-> Linear(out, bias). tfjs getWeights() order (what the JSON stores):
  k0 [in,h0], ln0_gamma [h0], ln0_beta [h0], k1 [h0,h1], ln1_gamma, ln1_beta,
  ..., k_out [hN,out], b_out [out]
Torch Linear.weight is [out,in], so kernels transpose on the way in/out.
"""

import numpy as np
import torch
import torch.nn as nn

LN_EPS = 1e-3


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim = in_dim, list(hidden), out_dim
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=False), nn.LayerNorm(h, eps=LN_EPS), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim, bias=True)
        # Match networks.mjs init: glorot-uniform hiddens, N(0, 0.01) output.
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(self.out.weight, 0.0, 0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.trunk(x))

    # --- tfjs-format (de)serialization --------------------------------------
    def to_records(self):
        recs = []
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                k = m.weight.detach().cpu().numpy().T  # [in, out]
                recs.append({"shape": list(k.shape), "data": k.flatten().tolist()})
            elif isinstance(m, nn.LayerNorm):
                g = m.weight.detach().cpu().numpy()
                b = m.bias.detach().cpu().numpy()
                recs.append({"shape": list(g.shape), "data": g.tolist()})
                recs.append({"shape": list(b.shape), "data": b.tolist()})
        k = self.out.weight.detach().cpu().numpy().T
        recs.append({"shape": list(k.shape), "data": k.flatten().tolist()})
        b = self.out.bias.detach().cpu().numpy()
        recs.append({"shape": list(b.shape), "data": b.tolist()})
        return recs

    def load_records(self, recs):
        arrays = [np.array(r["data"], dtype=np.float32).reshape(r["shape"]) for r in recs]
        i = 0
        with torch.no_grad():
            for m in self.trunk:
                if isinstance(m, nn.Linear):
                    m.weight.copy_(torch.from_numpy(arrays[i].T.copy()))
                    i += 1
                elif isinstance(m, nn.LayerNorm):
                    m.weight.copy_(torch.from_numpy(arrays[i]))
                    m.bias.copy_(torch.from_numpy(arrays[i + 1]))
                    i += 2
            self.out.weight.copy_(torch.from_numpy(arrays[i].T.copy()))
            self.out.bias.copy_(torch.from_numpy(arrays[i + 1]))

    @staticmethod
    def dims_from_records(recs):
        """Infer (in_dim, hidden, out_dim) from tfjs records (kernels are the
        [in,out]-shaped entries at stride 3, then the final kernel+bias)."""
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
