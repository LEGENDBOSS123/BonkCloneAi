"""Load a frozen actor from a ppo4 checkpoint.

ppo4 actors are always plain MLPs (tfjs record lists) — the recurrence lives
in the CRITIC, which never acts and is never pooled, so nothing that plays a
game here needs hidden state. The reset/reset_env hooks are kept so play.py
and eval_h2h.py read identically to the bonk2 versions.

    pol = load_policy(records)
    pol.reset(n_envs)
    acts = pol.act(states, greedy=True)
    pol.reset_env(i)                        # env i's episode ended (no-op)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .networks import MLP
from . import config as C


def is_recurrent(weights) -> bool:
    return isinstance(weights, dict)


class _Base:
    def reset(self, n: int):
        pass

    def reset_env(self, i: int):
        pass


class MLPPolicy(_Base):
    """Two-head (action, duration) actor with FiGAR holds. The wrapper owns the
    per-row hold state so play.py / eval_h2h call act() every decision-cycle and
    the hold is enforced transparently -- exactly as during training: it appends
    the normalised remaining-hold to the obs, re-samples only when the hold
    expires, and repeats the held action otherwise."""

    kind = "mlp"

    def __init__(self, records):
        self.net = MLP.from_records(records)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.state_dim = self.net.in_dim
        self.num_actions = C.NUM_ACTIONS
        self._durs = np.asarray(C.DURATIONS, dtype=np.int64)
        self._maxdur = float(max(C.DURATIONS))
        self.hold = None
        self.held_a = None
        self.hidden_size = 0

    def reset(self, n: int):
        self.hold = np.zeros(n, dtype=np.int64)
        self.held_a = np.zeros(n, dtype=np.int64)

    def reset_env(self, i: int):
        if self.hold is not None:
            self.hold[i] = 0            # new episode -> free next decision

    @torch.no_grad()
    def act(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        x = np.ascontiguousarray(states, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        n = x.shape[0]
        if self.hold is None or len(self.hold) != n:
            self.reset(n)
        hf = (self.hold.astype(np.float32) / self._maxdur)[:, None]
        logits = self.net(torch.from_numpy(np.concatenate([x, hf], 1)).float())
        la, ld = logits[:, :self.num_actions], logits[:, self.num_actions:]
        if greedy:
            a, d = la.argmax(1).numpy(), ld.argmax(1).numpy()
        else:
            a = torch.multinomial(torch.softmax(la, 1), 1).squeeze(1).numpy()
            d = torch.multinomial(torch.softmax(ld, 1), 1).squeeze(1).numpy()
        free = self.hold == 0
        self.held_a[free] = a[free]
        self.hold[free] = self._durs[d[free]]
        out = self.held_a.copy()
        self.hold -= 1
        return out


def load_policy(weights):
    """ppo4 actors are always plain MLPs — the recurrence lives in the critic,
    which never acts and is never pooled."""
    if isinstance(weights, dict) and not str(weights.get("format", "")).startswith("actor-v"):
        raise ValueError("this is a recurrent-ACTOR checkpoint (ppo3); ppo4 "
                         "actors are feedforward")
    return MLPPolicy(weights)


def resolve_agent(ckpt: dict, spec: str):
    """Pull one agent's weights + a label out of a full training checkpoint.
    spec: main | snap[:N] | exp[:N] (N indexes the pool, default -1 = newest).
    """
    kind, _, idx = spec.strip().lower().partition(":")
    agent = ckpt["agent"]
    if kind in ("main", "current", "actor"):
        w = agent.get("actor")
        if w is None:
            raise ValueError(f"no actor weights (agent keys: "
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
