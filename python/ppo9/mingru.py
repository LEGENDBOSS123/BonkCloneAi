"""minGRU recurrent encoder (Feng et al. 2024, "Were RNNs All We Needed?").

Chosen over GRU/LSTM (sequential BPTT, saturating gates -> unstable), LRU
(complex-valued, more bug surface) and a transformer (quadratic, heavy deploy):

* Its gates depend ONLY on the input, not on ``h``, so the recurrence is
  LINEAR: ``h_t = (1 - z_t) * h_{t-1} + z_t * h~_t`` with ``z_t = sigmoid(W_z
  x_t)``. Linear recurrence means stable gradients — no tanh saturation, no
  exploding or vanishing — which is exactly the "recurrent training is
  unstable" failure mode this avoids.
* ``z`` and ``h~`` for a whole sequence come out of one batched matmul; only
  the cheap elementwise scan is sequential, so a BPTT update stays fast.
* O(1) hidden-state step per decision at inference.

The net is minGRU -> MLP head with a SKIP CONNECTION: the MLP sees
``concat([x, h])``, so the head keeps direct access to the precise current
frame as well as the temporal summary.

Records are tfjs-compatible (dense kernels stored ``[in, out]``), so a trained
actor can be rebuilt in the browser.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

Records = dict[str, Any]


class MinGRUNet(nn.Module):
    """minGRU encoder + tanh MLP head.

    Shapes throughout: ``B`` batch, ``L`` sequence length, ``H`` = `gru_hidden`.
    """

    def __init__(self, in_dim: int, hidden: tuple[int, ...] | list[int],
                 out_dim: int, gru_hidden: int = 64) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = list(hidden)
        self.out_dim = out_dim
        self.H = int(gru_hidden)

        # One Linear produces [z | h_tilde] for the input-only gates.
        self.gru = nn.Linear(in_dim, 2 * self.H, bias=True)

        layers: list[nn.Module] = []
        d = in_dim + self.H                     # skip connection: [x || h]
        for h in self.hidden:
            layers += [nn.Linear(d, h, bias=True), nn.Tanh()]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim, bias=True)

        # Glorot on the tanh path; near-zero output so the initial policy is
        # ~uniform and the initial value ~0.
        nn.init.xavier_uniform_(self.gru.weight)
        nn.init.zeros_(self.gru.bias)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out.weight, 0.0, 0.01)
        nn.init.zeros_(self.out.bias)

    def zero_h(self, batch: int, device: torch.device | str | None = None) -> torch.Tensor:
        """``[batch, H]`` zeros on this net's device unless told otherwise."""
        return torch.zeros(batch, self.H, device=device or self.gru.weight.device)

    def _gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[..., in_dim]`` -> ``(z, h_tilde)``, each ``[..., H]``.

        Shape-agnostic in leading dims, so it serves both `step` (``[B, in]``)
        and `forward_seq` (``[B, L, in]``).
        """
        zt = self.gru(x)
        z, h_tilde = zt[..., :self.H], zt[..., self.H:]
        return torch.sigmoid(z), h_tilde

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One decision. ``x[B, in_dim]``, ``h[B, H]`` (previous).

        Returns ``(out[B, out_dim], h2[B, H])``. The head reads the POST-update
        hidden, so the output reflects the current input.
        """
        z, h_tilde = self._gates(x)
        h2 = (1.0 - z) * h + z * h_tilde
        return self.out(self.mlp(torch.cat([x, h2], dim=-1))), h2

    def step_trunk(self, x: torch.Tensor,
                   h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`step`, plus the pre-`out` trunk features ``[B, hidden[-1]]``.

        For an auxiliary head that needs the tail-state representation OUTSIDE
        a full-sequence BPTT pass — e.g. bootstrapping a multi-gamma value at
        the pending state, the same role the critic's own tail distribution
        plays for the categorical head.
        """
        z, h_tilde = self._gates(x)
        h2 = (1.0 - z) * h + z * h_tilde
        trunk = self.mlp(torch.cat([x, h2], dim=-1))
        return self.out(trunk), h2, trunk

    def forward_seq(self, x: torch.Tensor, h0: torch.Tensor | None = None,
                    reset: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Whole-sequence forward for the BPTT update.

        Args:
            x:     ``[B, L, in_dim]``.
            h0:    ``[B, H]`` incoming hidden; zeros if omitted.
            reset: ``[B, L]`` in {0, 1}. ``reset[b, t] == 1`` zeros the incoming
                   hidden BEFORE consuming ``x_t`` — it marks the first step of
                   a new episode inside the sequence.

        Returns:
            ``(out[B, L, out_dim], hs[B, L, H], trunk[B, L, hidden[-1]])``.
            `trunk` is exposed so auxiliary heads ride the SAME already-computed
            representation instead of paying for a second forward pass.
        """
        B, L, _ = x.shape
        z, h_tilde = self._gates(x)                 # both [B, L, H], one matmul
        h = self.zero_h(B, x.device) if h0 is None else h0
        hs = []
        for t in range(L):
            if reset is not None:
                h = h * (1.0 - reset[:, t:t + 1])   # new episode -> drop memory
            h = (1.0 - z[:, t]) * h + z[:, t] * h_tilde[:, t]
            hs.append(h)
        hs_t = torch.stack(hs, dim=1)               # [B, L, H]
        trunk = self.mlp(torch.cat([x, hs_t], dim=-1))
        return self.out(trunk), hs_t, trunk

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """Single-step convenience returning only the output.

        `h` defaults to zeros, i.e. MEMORYLESS. Used by shape checks, never in
        the collection or update loop.
        """
        h = self.zero_h(x.shape[0], x.device) if h is None else h
        return self.step(x, h)[0]

    # ── tfjs-compatible records ────────────────────────────────────────────
    @staticmethod
    def _dense(lin: nn.Linear) -> list[dict[str, Any]]:
        """One Linear as ``[kernel, bias]`` records, kernel TRANSPOSED to [in, out].

        ``[in, out]`` is the tfjs convention — the opposite of PyTorch's
        ``[out, in]``.
        """
        k = lin.weight.detach().cpu().numpy().T
        b = lin.bias.detach().cpu().numpy()
        return [{"shape": list(k.shape), "data": k.flatten().tolist()},
                {"shape": list(b.shape), "data": b.tolist()}]

    def to_records(self) -> Records:
        """Serialize to a JSON-safe dict.

        Note `self.out` is appended to the END of the flat ``"mlp"`` list
        rather than getting its own key — a reader must know the last pair is
        the output layer.
        """
        mlp: list[dict[str, Any]] = []
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                mlp += self._dense(m)
        mlp += self._dense(self.out)
        return {"type": "mingru",
                "meta": {"in_dim": self.in_dim, "hidden": self.hidden,
                         "out_dim": self.out_dim, "gru_hidden": self.H},
                "gru": self._dense(self.gru),
                "mlp": mlp}

    def load_records(self, recs: Records) -> None:
        """Load weights written by `to_records` (exact round-trip)."""
        def arr(r: dict[str, Any]) -> np.ndarray:
            return np.array(r["data"], dtype=np.float32).reshape(r["shape"])

        with torch.no_grad():
            # .copy() because .T yields a non-contiguous view from_numpy rejects.
            self.gru.weight.copy_(torch.from_numpy(arr(recs["gru"][0]).T.copy()))
            self.gru.bias.copy_(torch.from_numpy(arr(recs["gru"][1])))
            mlp, i = recs["mlp"], 0
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    m.weight.copy_(torch.from_numpy(arr(mlp[i]).T.copy()))
                    m.bias.copy_(torch.from_numpy(arr(mlp[i + 1])))
                    i += 2
            self.out.weight.copy_(torch.from_numpy(arr(mlp[i]).T.copy()))
            self.out.bias.copy_(torch.from_numpy(arr(mlp[i + 1])))

    @classmethod
    def from_records(cls, recs: Records) -> "MinGRUNet":
        """Rebuild a net from records, inferring its architecture from `meta`."""
        if recs.get("type") != "mingru":
            raise ValueError(f"not a mingru record set: type={recs.get('type')!r}")
        m = recs["meta"]
        net = cls(m["in_dim"], m["hidden"], m["out_dim"], gru_hidden=m["gru_hidden"])
        net.load_records(recs)
        return net
