"""Load a frozen ACTOR out of a training checkpoint and play with it.

`load_policy(records)` returns a wrapper that owns everything the training loop
does for a seat, so `play.py` and `eval_h2h.py` see the agent exactly as the
trainer did:

  * ARCHITECTURE is detected from the records themselves — a `type: "mingru"`
    dict rebuilds the recurrent net, `type: "temporal"` the conv net, a bare
    record list the legacy MLP. Callers never say which.
  * FiGAR HOLDS are enforced here: the wrapper appends the normalised remaining
    hold to the observation, re-samples only when a hold expires, and repeats
    the held action otherwise.
  * RECURRENT MEMORY is carried per env and MUST be dropped when an episode
    ends, or memory bleeds from one game into the next. That is what
    `reset_env(i)` is for — call it on every episode end.

    pol = load_policy(records)
    pol.reset(n_envs)
    acts = pol.act(states, greedy=True)     # states are STATE_DIM, not +hold
    pol.reset_env(i)                        # env i's episode ended

`resolve_agent(ckpt, spec)` picks WHICH agent out of a full training
checkpoint: the main, a snapshot, or a pool exploiter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .mingru import MinGRUNet
from .networks import MLP, TemporalNet


class _Base:
    def reset(self, n: int):
        pass

    def reset_env(self, i: int):
        pass


class ActorPolicy(_Base):
    """Two-head (action, duration) actor with FiGAR holds and, when the records
    say so, a carried minGRU hidden state.

    The hold state lives here so callers can just invoke act() once per
    decision-cycle and get training-identical behaviour: the normalised
    remaining hold is appended to the obs, a new action is sampled only when the
    hold expires, and the held action is repeated otherwise. The recurrent
    hidden state is stepped EVERY cycle regardless of the hold, matching the
    trainer (the hold gates the action, not the memory).
    """

    kind = "mlp"

    def __init__(self, records):
        recs = records.get("records", records) if isinstance(records, dict) else records
        self.recurrent = isinstance(recs, dict) and recs.get("type") == "mingru"
        if self.recurrent:
            self.net = MinGRUNet.from_records(recs)
        elif isinstance(recs, dict) and recs.get("type") == "temporal":
            self.net = TemporalNet.from_records(recs)
        else:
            self.net = MLP.from_records(recs)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.state_dim = self.net.in_dim
        self.num_actions = C.NUM_ACTIONS
        self._durs = np.asarray(C.DURATIONS, dtype=np.int64)
        self._maxdur = float(max(C.DURATIONS))
        self.hold = None
        self.held_a = None
        self.h = None                    # recurrent hidden state [n, H]
        self.hidden_size = C.MINGRU_HIDDEN if self.recurrent else 0

    def reset(self, n: int):
        self.hold = np.zeros(n, dtype=np.int64)
        self.held_a = np.zeros(n, dtype=np.int64)
        if self.recurrent:
            self.h = self.net.zero_h(n)

    def reset_env(self, i: int):
        if self.hold is not None:
            self.hold[i] = 0            # new episode -> free next decision
        if self.recurrent and self.h is not None:
            self.h[i] = 0.0             # new episode -> drop memory

    @torch.no_grad()
    def act(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        x = np.ascontiguousarray(states, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        n = x.shape[0]
        if self.hold is None or len(self.hold) != n:
            self.reset(n)
        hf = (self.hold.astype(np.float32) / self._maxdur)[:, None]
        inp = torch.from_numpy(np.concatenate([x, hf], 1)).float()
        if self.recurrent:
            logits, self.h = self.net.step(inp, self.h)   # step hidden every cycle
        else:
            logits = self.net(inp)
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
    """Build a playable policy from whatever an actor was saved as: a
    `type: "mingru"` dict (recurrent), a `type: "temporal"` dict (conv), a flat
    dense record list (legacy MLP), or an exported actor blob wrapping any of
    them. An unrecognised dict is rejected rather than silently mis-loaded."""
    w = weights
    if isinstance(w, dict):
        ok = (w.get("type") in ("temporal", "mingru")
              or str(w.get("format", "")).startswith("actor-v")
              or "records" in w)
        if not ok:
            raise ValueError(f"unrecognised actor records (keys: {sorted(w)}); "
                             f"expected type mingru/temporal, an actor-v blob, "
                             f"or a flat dense record list")
    return ActorPolicy(weights)


# Back-compat alias: this class was called MLPPolicy when every actor was an MLP.
MLPPolicy = ActorPolicy


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
