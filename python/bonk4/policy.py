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
        from .actor import ActorNet
        self.net = (ActorNet.from_records(records)
                    if isinstance(records, dict) and
                       records.get('format') == 'actor-v2'
                    else MLP.from_records(records))
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.state_dim = getattr(self.net, 'in_dim', None) or self.net.state_dim
        self.num_actions = getattr(self.net, 'out_dim', None) or self.net.num_actions
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


def load_policy(weights):
    """ppo4 actors are always plain MLPs — the recurrence lives in the critic,
    which never acts and is never pooled."""
    if isinstance(weights, dict) and weights.get("format") != "actor-v2":
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
