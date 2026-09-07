"""minGRU recurrent encoder (Feng et al. 2024, "Were RNNs All We Needed?").

Why minGRU over GRU / LSTM / LRU / Transformer here:
  * Its gates depend ONLY on the input (not on h), so the recurrence is LINEAR:
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    with z_t = sigmoid(W_z x_t), h_tilde_t = W_h x_t. Linear recurrence => stable
    gradients (no tanh saturation, no exploding/vanishing), which is exactly the
    "recurrent training is unstable" failure mode we wanted to avoid.
  * z_t and h_tilde_t are computed for the whole sequence in one batched matmul
    (parallel); only the cheap elementwise scan is sequential. So a burn-in +
    BPTT update stays fast.
  * O(1) hidden-state step per decision at inference (vs the TCN's O(window)).

Net = minGRU encoder -> MLP head, with a skip connection (the current input is
concatenated with the recurrent memory before the MLP) so the head keeps direct
access to the precise current frame in addition to the temporal summary. Same
tanh-MLP conventions and tfjs-compatible dense records as networks.MLP, plus the
two GRU weight matrices, so eval/deploy rebuild from records.

Interfaces:
  step(x[B,in], h[B,H])            -> (out[B,out], h2[B,H])          # collection
  forward_seq(x[B,L,in], h0, reset)-> (out[B,L,out], hs[B,L,H])      # update (BPTT)
`reset[b,t]==1` zeros the incoming hidden BEFORE consuming x_t, i.e. it marks the
first step of a new episode inside a sequence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MinGRUNet(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int,
                 gru_hidden: int = 64):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim = in_dim, list(hidden), out_dim
        self.H = int(gru_hidden)
        # minGRU gates (input-only). One Linear producing [z | h_tilde] is simplest.
        self.gru = nn.Linear(in_dim, 2 * self.H, bias=True)
        # MLP head over [current input || recurrent memory] (skip connection).
        layers, d = [], in_dim + self.H
        for h in self.hidden:
            layers += [nn.Linear(d, h, bias=True), nn.Tanh()]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim, bias=True)
        # init: glorot on tanh path; near-zero output => ~uniform policy / ~0 value.
        nn.init.xavier_uniform_(self.gru.weight)
        nn.init.zeros_(self.gru.bias)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out.weight, 0.0, 0.01)
        nn.init.zeros_(self.out.bias)

    def zero_h(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.H, device=device or self.gru.weight.device)

    def _gates(self, x):
        """z, h_tilde for x[..., in] -> each [..., H]."""
        zt = self.gru(x)
        z, h_tilde = zt[..., :self.H], zt[..., self.H:]
        return torch.sigmoid(z), h_tilde

    def _head(self, x, h):
        return self.out(self.mlp(torch.cat([x, h], dim=-1)))

    def step(self, x, h):
        """One recurrent step. x[B,in], h[B,H] (previous) -> (out[B,out], h2[B,H])."""
        z, h_tilde = self._gates(x)
        h2 = (1.0 - z) * h + z * h_tilde
        return self._head(x, h2), h2

    def step_trunk(self, x, h):
        """Like `step`, but also returns the pre-`out` trunk features (see
        `forward_seq`'s `trunk`) -- for an auxiliary head that needs the
        tail-state representation OUTSIDE a full-sequence BPTT pass, e.g.
        bootstrapping AUX_GAMMA's value at the pending state (the same role
        `_tail_dist` plays for the categorical critic, one level lower)."""
        z, h_tilde = self._gates(x)
        h2 = (1.0 - z) * h + z * h_tilde
        trunk = self.mlp(torch.cat([x, h2], dim=-1))
        return self.out(trunk), h2, trunk

    def forward_seq(self, x, h0=None, reset=None):
        """x[B,L,in]; h0[B,H] (default zeros); reset[B,L] in {0,1} zeroing the
        incoming hidden before each flagged step. Returns
        (out[B,L,out], hs[B,L,H], trunk[B,L,hidden[-1]]) -- `trunk` is the
        post-MLP, pre-`out` features, exposed for auxiliary heads that want to
        ride on the SAME already-computed representation (RUDDER/aux_head/
        time_head-style) rather than paying for a second forward pass."""
        B, L, _ = x.shape
        z, h_tilde = self._gates(x)                 # [B,L,H] each (parallel)
        h = self.zero_h(B, x.device) if h0 is None else h0
        hs = []
        for t in range(L):
            if reset is not None:
                h = h * (1.0 - reset[:, t:t + 1])   # new episode -> drop memory
            h = (1.0 - z[:, t]) * h + z[:, t] * h_tilde[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)                 # [B,L,H]
        trunk = self.mlp(torch.cat([x, hs], dim=-1))
        out = self.out(trunk)
        return out, hs, trunk

    def forward(self, x, h=None):
        """Convenience single-step forward returning only the output (h defaults
        to zeros -> memoryless; used only by shape checks, never in the loop)."""
        h = self.zero_h(x.shape[0], x.device) if h is None else h
        return self.step(x, h)[0]

    # ── tfjs-compatible records: dense kernels stored [in,out] ──────────────────
    @staticmethod
    def _dense(lin):
        k = lin.weight.detach().cpu().numpy().T
        b = lin.bias.detach().cpu().numpy()
        return [{"shape": list(k.shape), "data": k.flatten().tolist()},
                {"shape": list(b.shape), "data": b.tolist()}]

    def to_records(self):
        mlp = []
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                mlp += self._dense(m)
        mlp += self._dense(self.out)
        return {
            "type": "mingru",
            "meta": {"in_dim": self.in_dim, "hidden": self.hidden,
                     "out_dim": self.out_dim, "gru_hidden": self.H},
            "gru": self._dense(self.gru), "mlp": mlp,
        }

    def load_records(self, recs):
        def arr(r):
            return np.array(r["data"], dtype=np.float32).reshape(r["shape"])
        with torch.no_grad():
            self.gru.weight.copy_(torch.from_numpy(arr(recs["gru"][0]).T.copy()))
            self.gru.bias.copy_(torch.from_numpy(arr(recs["gru"][1])))
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
        net = cls(m["in_dim"], m["hidden"], m["out_dim"], gru_hidden=m["gru_hidden"])
        net.load_records(recs)
        return net
