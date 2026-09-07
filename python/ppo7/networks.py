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
import torch.nn.functional as F

from . import config as C


class TemporalNet(nn.Module):
    """Dilated causal 1-D conv (TCN) over a RAW-frame window + the existing MLP.

    Input layout (per config): x = [ raw window (TCN_WINDOW x RAW_FRAME_DIM,
    NEWEST-FIRST) || Fourier(current frame) (FOURIER_BLOCK) || extra (in_dim -
    STATE_DIM, e.g. the FiGAR hold) ]. The conv reads the raw window as a
    sequence and emits a fixed embedding; the MLP then sees exactly the SAME
    inputs it saw before (current raw + current Fourier + extra) PLUS that
    embedding -- strictly additive. It is a pure feedforward function of the
    current windowed input (no hidden state carried across the rollout), so PPO
    / GAE / bootstrapping are unchanged. Same trunk/out/records interface as MLP,
    so the rest of the agent, eval and deploy are untouched aside from the shape.
    """

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim = in_dim, list(hidden), out_dim
        self.W, self.R, self.FB = C.TCN_WINDOW, C.RAW_FRAME_DIM, C.FOURIER_BLOCK
        self.n_extra = in_dim - C.STATE_DIM        # appended features (hold)
        self.kernel = C.TCN_KERNEL
        self.dilations = list(C.TCN_DILATIONS)
        ch = C.TCN_CHANNELS
        # dilated causal conv stack (kernel weights shared across time)
        self.convs = nn.ModuleList()
        cin = self.R
        for dil in self.dilations:
            self.convs.append(nn.Conv1d(cin, ch, self.kernel, dilation=dil))
            cin = ch
        self.embed = nn.Linear(ch, C.TCN_EMBED)
        # trunk input = current raw (R) + current Fourier (FB) + extra + embedding
        trunk_in = self.R + self.FB + self.n_extra + C.TCN_EMBED
        layers, d = [], trunk_in
        for h in self.hidden:
            layers += [nn.Linear(d, h, bias=True), nn.Tanh()]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim, bias=True)
        # init: glorot for tanh path, near-zero output so the initial policy is
        # ~uniform / the initial value ~0 (matches MLP).
        for m in list(self.convs) + [self.embed] + list(self.mlp):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out.weight, 0.0, 0.01)
        nn.init.zeros_(self.out.bias)

    def _encode(self, x):
        """x[B,in_dim] -> trunk input [B, R+FB+n_extra+EMBED]."""
        B = x.shape[0]
        win = x[:, : self.W * self.R].reshape(B, self.W, self.R)   # newest-first
        fourier = x[:, self.W * self.R : self.W * self.R + self.FB]
        cur_raw = win[:, 0, :]                                     # newest frame
        # chronological (oldest->newest) so the LAST conv output is "now"
        h = win.flip(1).transpose(1, 2).contiguous()             # [B, R, W]
        # Dilated causal conv as batched matmul per tap. Mathematically identical
        # to nn.Conv1d with left-pad (k-1)*dil (verified ~1e-6), but ~15x faster
        # on CPU: conv1d's im2col dispatch dominates for a tiny kernel and even
        # SLOWS with more threads, while einsum stays a couple of GEMMs.
        L = h.shape[-1]
        for conv, dil in zip(self.convs, self.dilations):
            Wt, b = conv.weight, conv.bias                       # [out, in, k]
            acc = b[None, :, None]
            for j in range(self.kernel):                         # tap j -> x[t-(k-1-j)*dil]
                shift = (self.kernel - 1 - j) * dil
                xj = h if shift == 0 else F.pad(h, (shift, 0))[..., :L]
                acc = acc + torch.einsum('oi,bil->bol', Wt[:, :, j], xj)
            y = torch.tanh(acc)
            h = y + h if y.shape == h.shape else y               # residual when Ch matches
        emb = torch.tanh(self.embed(h[:, :, -1]))                 # newest step
        parts = [cur_raw, fourier]
        if self.n_extra > 0:
            parts.append(x[:, C.STATE_DIM:])
        parts.append(emb)
        return torch.cat(parts, dim=1)

    def trunk(self, x):
        return self.mlp(self._encode(x))

    def forward(self, x):
        return self.out(self.trunk(x))

    # --- records: {meta, conv[], embed[], mlp[]}; mlp[] is the tfjs dense format -
    def _dense_recs(self, linear):
        k = linear.weight.detach().cpu().numpy().T            # [in, out]
        b = linear.bias.detach().cpu().numpy()
        return [{"shape": list(k.shape), "data": k.flatten().tolist()},
                {"shape": list(b.shape), "data": b.tolist()}]

    def to_records(self):
        conv = []
        for cv in self.convs:
            w = cv.weight.detach().cpu().numpy()              # [out, in, k] (torch)
            b = cv.bias.detach().cpu().numpy()
            conv += [{"shape": list(w.shape), "data": w.flatten().tolist()},
                     {"shape": list(b.shape), "data": b.tolist()}]
        mlp = []
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                mlp += self._dense_recs(m)
        mlp += self._dense_recs(self.out)
        return {
            "type": "temporal",
            "meta": {"window": self.W, "raw": self.R, "fourier": self.FB,
                     "n_extra": self.n_extra, "channels": C.TCN_CHANNELS,
                     "kernel": self.kernel, "dilations": self.dilations,
                     "embed": C.TCN_EMBED, "in_dim": self.in_dim,
                     "hidden": self.hidden, "out_dim": self.out_dim},
            "conv": conv, "embed": self._dense_recs(self.embed), "mlp": mlp,
        }

    def load_records(self, recs):
        def arr(r):
            return np.array(r["data"], dtype=np.float32).reshape(r["shape"])
        with torch.no_grad():
            for i, cv in enumerate(self.convs):
                cv.weight.copy_(torch.from_numpy(arr(recs["conv"][2 * i]).copy()))
                cv.bias.copy_(torch.from_numpy(arr(recs["conv"][2 * i + 1])))
            self.embed.weight.copy_(torch.from_numpy(arr(recs["embed"][0]).T.copy()))
            self.embed.bias.copy_(torch.from_numpy(arr(recs["embed"][1])))
            mlp = recs["mlp"]; i = 0
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    m.weight.copy_(torch.from_numpy(arr(mlp[i]).T.copy()))
                    m.bias.copy_(torch.from_numpy(arr(mlp[i + 1])))
                    i += 2
            self.out.weight.copy_(torch.from_numpy(arr(mlp[i]).T.copy()))
            self.out.bias.copy_(torch.from_numpy(arr(mlp[i + 1])))

    @classmethod
    def from_records(cls, recs):
        m = recs["meta"]
        net = cls(m["in_dim"], m["hidden"], m["out_dim"])
        net.load_records(recs)
        return net


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
