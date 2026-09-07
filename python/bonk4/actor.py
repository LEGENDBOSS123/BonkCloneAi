"""Actor with the transplanted movement CORE.

    obs[34] ─► trunk 34→256→256 ─────────────────────────┐  h_task[256]
                     │                                    │
                     └─► proj_in 256→64 ─► emb_norm = u   │
                                │                          │
                         CORE 64→64→64  (frozen)           │
                                │  m[64]                   │
                                └────────► concat[320] ◄───┘
                                                │
                                          fuse 320→256 = z
                                                │
                                          out 256→18

The CORE arrives pre-trained on a dense single-agent goal-reaching task (see
pretrain_move.py) and is FROZEN: PPO gradients reach `proj_in` through it but
never change its weights. The actor therefore learns what to *ask* the movement
module, not how to rebuild it.

emb_norm is what makes the transplant meaningful. Left unconstrained, proj_in
can drive the CORE anywhere in R^64 -- including regions where its learned
structure means nothing -- and the CORE degenerates into an arbitrary fixed
nonlinearity, i.e. no better than an ordinary layer. Normalising to the
embedding statistics recorded during pre-training keeps it on the manifold it
actually understands, while leaving the actor free to choose a direction.

`z` (not h_task) feeds the critic, so the value function sees exactly the
features the policy acts on.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .movement import MovementCore

LN_EPS = 1e-3


class ActorNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden: list[int] | None = None, d_emb: int | None = None):
        super().__init__()
        hid = list(hidden or C.HIDDEN)
        d = d_emb or C.MOVE_EMB
        self.state_dim, self.num_actions = state_dim, num_actions
        self.hidden, self.d_emb = hid, d

        layers, prev = [], state_dim
        for w in hid:
            layers += [nn.Linear(prev, w, bias=False),
                       nn.LayerNorm(w, eps=LN_EPS), nn.ReLU()]
            prev = w
        self.trunk = nn.Sequential(*layers)
        self.z_dim = prev

        self.use_core = bool(C.MOVE_TRANSPLANT)
        if self.use_core:
            self.proj_in = nn.Linear(prev, d)
            self.pre_norm = nn.LayerNorm(d, eps=LN_EPS, elementwise_affine=False)
            self.core = MovementCore(d, C.MOVE_HIDDEN)
            for p in self.core.parameters():      # frozen: PPO must not reshape it
                p.requires_grad_(False)
            self.fuse = nn.Sequential(
                nn.Linear(prev + d, prev), nn.LayerNorm(prev, eps=LN_EPS), nn.ReLU())
            self.register_buffer("emb_mean", torch.zeros(d))
            self.register_buffer("emb_std", torch.ones(d))

        self.out = nn.Linear(prev, num_actions)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(self.out.weight, 0.0, 0.01)
        nn.init.zeros_(self.out.bias)

    # ── forward ────────────────────────────────────────────────────────────
    def features(self, x):
        """-> z, the representation both the policy head and the critic read."""
        h = self.trunk(x)
        if not self.use_core:
            return h
        u = self.pre_norm(self.proj_in(h)) * self.emb_std + self.emb_mean
        return self.fuse(torch.cat([h, self.core(u)], dim=-1))

    def forward(self, x):
        return self.out(self.features(x))

    def core_input(self, x):
        """u, for monitoring: if std(u) collapses the branch is dead weight."""
        return self.pre_norm(self.proj_in(self.trunk(x))) * self.emb_std + self.emb_mean

    # ── pre-trained core ───────────────────────────────────────────────────
    def load_core(self, path: str) -> bool:
        if not self.use_core:
            return False
        with open(path) as f:
            obj = json.load(f)
        if obj.get("d_emb") != self.d_emb:
            print(f"note: move core d_emb {obj.get('d_emb')} != {self.d_emb}; "
                  f"skipping transplant")
            return False
        self.core.load_state_dict({k: torch.tensor(v)
                                   for k, v in obj["core"].items()})
        with torch.no_grad():
            self.emb_mean.copy_(torch.tensor(obj["emb_mean"]))
            self.emb_std.copy_(torch.tensor(obj["emb_std"]).clamp_min(1e-3))
        for p in self.core.parameters():
            p.requires_grad_(False)
        return True

    # ── persistence ────────────────────────────────────────────────────────
    def to_records(self) -> dict:
        return {"format": "actor-v2", "stateDim": self.state_dim,
                "numActions": self.num_actions, "hidden": self.hidden,
                "dEmb": self.d_emb, "useCore": self.use_core,
                "tensors": {k: v.detach().cpu().numpy().tolist()
                            for k, v in self.state_dict().items()}}

    def load_records(self, obj):
        if not isinstance(obj, dict) or obj.get("format") != "actor-v2":
            raise ValueError("not an actor-v2 record (pre-transplant "
                             "checkpoints are not loadable into this actor)")
        sd = {k: torch.tensor(v) for k, v in obj["tensors"].items()}
        own = self.state_dict()
        drop = [k for k in sd if k not in own or sd[k].shape != own[k].shape]
        for k in drop:
            del sd[k]
        if drop:
            print(f"note: actor reinitialising {', '.join(drop)}")
            for k in drop:
                sd[k] = own[k]
        self.load_state_dict(sd)

    @classmethod
    def from_records(cls, obj):
        net = cls(obj["stateDim"], obj["numActions"], obj["hidden"], obj["dEmb"])
        net.load_records(obj)
        return net
