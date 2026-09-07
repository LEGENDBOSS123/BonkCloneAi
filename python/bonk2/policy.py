"""Load a frozen policy from a checkpoint, feedforward or recurrent.

The architecture is detected from the WEIGHTS, not from config.RECURRENT: a
checkpoint has to load as whatever it was saved as, and `play`/`eval_h2h` are
inspecting artefacts rather than continuing a run. config.RECURRENT governs
training only.

  feedforward -> tfjs record LIST  (MLP.from_records infers the shape)
  recurrent   -> dict {format: "gru-v1", tensors: {...}}

Both expose the same batched surface, with hidden state owned by the policy so
callers only have to say when an episode ended:

    pol = load_policy(records)
    pol.reset(n_envs)
    acts = pol.act(states, greedy=True)     # advances hidden state
    pol.reset_env(i)                        # env i's episode ended
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from bonk.networks import MLP


def is_recurrent(weights) -> bool:
    return isinstance(weights, dict)


class _Base:
    def reset(self, n: int):
        pass

    def reset_env(self, i: int):
        pass


class MLPPolicy(_Base):
    kind = "mlp"

    def __init__(self, records):
        self.net = MLP.from_records(records)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.state_dim = self.net.in_dim
        self.num_actions = self.net.out_dim
        self.hidden_size = 0

    @torch.no_grad()
    def act(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(states)).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        logits = self.net(x)
        a = (logits.argmax(1) if greedy
             else torch.multinomial(torch.softmax(logits, 1), 1).squeeze(1))
        return a.numpy()


class GRUPolicy(_Base):
    kind = "gru"

    def __init__(self, obj):
        from .recurrent import RecurrentAC
        self.net = RecurrentAC(obj["stateDim"], obj["numActions"], obj["hidden"])
        self.net.load_records(obj)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.state_dim = obj["stateDim"]
        self.num_actions = obj["numActions"]
        self.hidden_size = obj["hidden"]
        self.h = torch.zeros(1, self.hidden_size)

    def reset(self, n: int):
        self.h = torch.zeros(n, self.hidden_size)

    def reset_env(self, i: int):
        """An episode ended: that env's memory must not leak into the next one —
        the same rule the trainer's done-mask enforces."""
        self.h[i].zero_()

    @torch.no_grad()
    def act(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(states)).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if self.h.shape[0] != x.shape[0]:
            self.reset(x.shape[0])
        logits, _v, self.h = self.net.step(x, self.h)
        a = (logits.argmax(1) if greedy
             else torch.multinomial(F.softmax(logits, 1), 1).squeeze(1))
        return a.numpy()


def load_policy(weights):
    return GRUPolicy(weights) if is_recurrent(weights) else MLPPolicy(weights)


def resolve_agent(ckpt: dict, spec: str):
    """Pull one agent's weights + a label out of a full training checkpoint.
    spec: main | snap[:N] | exp[:N] (N indexes the pool, default -1 = newest).
    Works for both architectures — a recurrent save stores the whole net under
    `agent.recurrent` rather than an `actor`."""
    kind, _, idx = spec.strip().lower().partition(":")
    agent = ckpt["agent"]
    if kind in ("main", "current", "actor"):
        w = agent.get("recurrent") or agent.get("actor")
        if w is None:
            raise ValueError(f"no actor/recurrent weights (agent keys: "
                             f"{sorted(agent)})")
        return w, "main"
    if kind in ("snap", "snapshot"):
        pool = ckpt.get("snapshots", [])
        if not pool:
            raise ValueError("checkpoint has no snapshots")
        i = int(idx) if idx else -1
        it = pool[i]
        return it["weights"], f"snap[{i % len(pool)}] ep{it.get('episode', '?')}"
    if kind in ("exp", "exploiter"):
        pool = ckpt.get("league", {}).get("exploiters", [])
        if not pool:
            raise ValueError("checkpoint has no exploiters")
        i = int(idx) if idx else -1
        it = pool[i]
        return (it["weights"],
                f"exp[{i % len(pool)}] ep{it.get('episode', '?')} "
                f"wr{it.get('winrate', 0.0):.2f}")
    raise ValueError(f"bad opponent spec: {spec!r} (want main | snap[:N] | exp[:N])")
