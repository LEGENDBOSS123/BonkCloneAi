"""Frozen actor for evaluation and play.

Rebuilds an actor from checkpoint records and reproduces TRAINING behaviour
exactly: the normalized remaining hold is appended to the observation, a new
action is sampled only when the hold expires, and the recurrent hidden state is
stepped EVERY cycle regardless of the hold (the hold gates the action, not the
memory). Getting that wrong makes eval quietly disagree with training.
"""
from __future__ import annotations

import glob
import json
from typing import Any

import numpy as np
import torch

from bonkenv import NUM_ACTIONS
from .config import FigarConfig
from .mingru import MinGRUNet, Records


class ActorPolicy:
    """A two-head (action, duration) actor with FiGAR holds and carried memory."""

    def __init__(self, records: Records,
                 figar: FigarConfig | None = None) -> None:
        self.net = MinGRUNet.from_records(records)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

        figar = figar or FigarConfig()
        self.state_dim = self.net.in_dim
        self.num_actions = NUM_ACTIONS
        self._durs = np.asarray(figar.durations, dtype=np.int64)
        self._maxdur = float(max(figar.durations))
        self.hold: np.ndarray | None = None
        self.held_a: np.ndarray | None = None
        self.h: torch.Tensor | None = None

    def reset(self, n: int) -> None:
        """Size the policy for `n` parallel environments and clear all state."""
        self.hold = np.zeros(n, dtype=np.int64)
        self.held_a = np.zeros(n, dtype=np.int64)
        self.h = self.net.zero_h(n)

    def reset_env(self, i: int) -> None:
        """One env's episode ended: free its next decision and drop its memory.

        A recurrent policy MUST have this called, or memory bleeds across games
        and the evaluation silently measures something else.
        """
        if self.hold is not None:
            self.hold[i] = 0
        if self.h is not None:
            self.h[i] = 0.0

    @torch.no_grad()
    def act(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        """``[n, state_dim]`` observations -> ``[n]`` action indices to apply."""
        x = np.ascontiguousarray(states, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        n = x.shape[0]
        if self.hold is None or len(self.hold) != n:
            self.reset(n)

        hf = (self.hold.astype(np.float32) / self._maxdur)[:, None]
        inp = torch.from_numpy(np.concatenate([x, hf], 1)).float()
        logits, self.h = self.net.step(inp, self.h)      # memory steps every cycle
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


def resolve_agent(ckpt: dict[str, Any], spec: str) -> tuple[Records, str]:
    """Pick one actor out of a checkpoint.

    Args:
        ckpt: a loaded ppo9 checkpoint.
        spec: ``"main"``, ``"snap:N"`` or ``"exp:N"`` (N is 1-based; negative
              counts from the newest, so ``snap:-1`` is the latest snapshot).

    Returns:
        ``(records, label)``.
    """
    if spec == "main":
        return ckpt["agent"]["actor"], "main"
    kind, _, raw = spec.partition(":")
    pool = (ckpt.get("snapshots", []) if kind == "snap"
            else ckpt.get("league", {}).get("exploiters", []))
    if not pool:
        raise ValueError(f"checkpoint has no {kind} pool")
    i = int(raw) if raw else -1
    idx = (i - 1) if i > 0 else (len(pool) + i)
    if not 0 <= idx < len(pool):
        raise ValueError(f"{spec}: index out of range (pool has {len(pool)})")
    return pool[idx]["weights"], f"{kind}{idx + 1}/{len(pool)}"


def load_checkpoint(path_glob: str) -> tuple[dict[str, Any], str]:
    """Load the newest checkpoint matching a glob. Returns ``(obj, path)``."""
    matches = sorted(glob.glob(path_glob))
    if not matches:
        raise FileNotFoundError(path_glob)
    path = matches[-1]
    with open(path) as f:
        return json.load(f), path


def load_policy(records: Records, figar: FigarConfig | None = None) -> ActorPolicy:
    """Build an `ActorPolicy` from records."""
    return ActorPolicy(records, figar)
