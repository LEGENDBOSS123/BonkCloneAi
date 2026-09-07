"""Pure array -> array learning targets. No torch, no config, no trainer state.

These are the parts of the update most likely to be SILENTLY wrong — an
off-by-one across an episode boundary, a target built from the wrong seat — and
none of it shows up in a training curve. Keeping them pure means they can be
pinned against a reference implementation in milliseconds
(`tests/test_targets.py`).

Every scalar arrives as an argument rather than being read from a config
module, so a test can vary one without touching global state.
"""
from __future__ import annotations

import numpy as np


def gae_columns(r: np.ndarray, v: np.ndarray, d: np.ndarray, last_v: np.ndarray,
                gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation, vectorized across environments.

    Args:
        r:      ``[T, E]`` rewards.
        v:      ``[T, E]`` critic values at each row.
        d:      ``[T, E]`` done flags (0/1).
        last_v: ``[E]`` bootstrap value for a cut-off (non-terminal) tail.
        gamma:  discount. 1.0 under the categorical critic.
        gae_lambda: trace decay.

    Returns:
        ``(advantages[T, E], returns[T, E])`` where ``returns = adv + v``.
    """
    T = r.shape[0]
    adv = np.zeros_like(r)
    acc = np.zeros(r.shape[1], np.float32)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - d[t]
        next_v = last_v if t == T - 1 else v[t + 1]
        delta = r[t] + gamma * next_v * nonterminal - v[t]
        acc = delta + gamma * gae_lambda * nonterminal * acc
        adv[t] = acc
    return adv, adv + v


def outcome_targets(dones: np.ndarray, outcome: np.ndarray,
                    tail_dist: np.ndarray, n_atoms: int = 3) -> np.ndarray:
    """``[T, E, n_atoms]`` target distribution over the terminal outcome.

    With gamma = 1 and terminal-only rewards the distributional Bellman backup
    is ``Z(s_t) = Z(s_{t+1})`` away from terminal, so the target simply
    propagates BACKWARD from each episode end.

    Rows whose episode has not resolved inside the window inherit `tail_dist`,
    the critic's own (detached) prediction at the pending state — which is what
    lets the update fire on a fixed number of STEPS instead of waiting for
    whole episodes to finish.

    Args:
        dones:     ``[T, E]`` done flags.
        outcome:   ``[T, E]`` int64 class (0=win, 1=draw, 2=loss; -1 = running).
                   The ACTUAL result — never nearest-atom-to-reward, which
                   merges draw and loss whenever their rewards match.
        tail_dist: ``[E, n_atoms]`` bootstrap distribution for unresolved tails.
        n_atoms:   size of the outcome support.

    Returns:
        ``[T, E, n_atoms]`` float32. Unlike ppo7 this is NOT pre-flattened;
        flattening is the caller's business.
    """
    T, E = dones.shape
    tgt = np.zeros((T, E, n_atoms), np.float32)
    cur = tail_dist.astype(np.float32).copy()
    for t in range(T - 1, -1, -1):
        d = dones[t] > 0
        if d.any():
            cls = outcome[t][d]
            oh = np.zeros((int(d.sum()), n_atoms), np.float32)
            oh[np.arange(len(cls)), cls] = 1.0
            cur = cur.copy()
            cur[d] = oh
        tgt[t] = cur
    return tgt


def multi_gamma_targets(dones: np.ndarray, outcome: np.ndarray,
                        gammas: tuple[float, ...], atom_vals: np.ndarray,
                        tail_values: np.ndarray | None = None) -> np.ndarray:
    """``[T, E, G]`` discounted value targets for the auxiliary gamma heads.

    Same backward propagation as `outcome_targets`, but each gamma discounts
    the realized outcome value by ``gamma ** k``, where ``k = 0`` at the row
    that RESOLVES the episode (undiscounted there, matching `outcome_targets`
    exactly at gamma = 1) and increases by one for every row further back. That
    is the finite-horizon fixed point of ``V_g(s) = g * V_g(s')`` under a
    terminal-only reward.

    Args:
        dones:       ``[T, E]`` done flags.
        outcome:     ``[T, E]`` int64 class, -1 where still running.
        gammas:      ``G`` discount factors.
        atom_vals:   ``[3]`` values ordered like `outcome`. Must be the OWNING
                     AGENT's atoms — an exploiter values a draw differently.
        tail_values: ``[E, G]`` bootstrap for unresolved tails (the head's own
                     detached prediction at the pending state); zeros if None.

    Returns:
        ``[T, E, G]`` float32.
    """
    T, E = dones.shape
    G = len(gammas)
    gam = np.asarray(gammas, np.float32)
    atoms = np.asarray(atom_vals, np.float32)
    tgt = np.zeros((T, E, G), np.float32)
    cur = (np.zeros((E, G), np.float32) if tail_values is None
           else tail_values.astype(np.float32).copy())
    k = np.zeros(E, np.float32)         # decisions since resolution
    for t in range(T - 1, -1, -1):
        d = dones[t] > 0
        if d.any():
            k = k.copy(); k[d] = 0.0
            cur = cur.copy(); cur[d] = atoms[outcome[t][d]][:, None]
        tgt[t] = cur * (gam[None, :] ** k[:, None])
        k = k + 1.0
    return tgt


def resolved_mask(dones: np.ndarray) -> np.ndarray:
    """``[T, E]`` bool: this row belongs to an episode that ENDS in the window.

    Rows in an unresolved trailing segment are False. Used to restrict a check
    or a loss to rows carrying ground truth rather than a bootstrap.
    """
    T, E = dones.shape
    out = np.zeros((T, E), bool)
    seen = np.zeros(E, bool)
    for t in range(T - 1, -1, -1):
        seen = seen | (dones[t] > 0)
        out[t] = seen
    return out


def approach_shaping(buf_s: np.ndarray, buf_d: np.ndarray, T: int, E: int,
                     rel_off: int, pos_scale: float, d_norm: float) -> np.ndarray:
    """``[T, E]`` dense reward for CLOSING DISTANCE, for the phase-1 warm start.

    Rewards the RISE in a bounded closeness potential from this decision to the
    next, not the level: getting closer pays, standing still pays zero, backing
    off pays negative. An occupancy reward instead makes the agent park in a
    decent spot and collect — measured, it stands still — because moving on
    fragile terrain risks a fatal fall. Rewarding current closeness gives no
    gradient at all, since a far agent never samples close states (measured
    0.2% contact over 500k episodes).

    Deliberately NOT policy-invariant: it exists to bias the from-scratch actor
    toward engagement, and is annealed fully out before the league runs.

    Args:
        buf_s:     ``[>=T, E, D]`` observations; ``[..., rel_off:rel_off+2]``
                   is the POS_SCALEd relative position.
        buf_d:     ``[>=T, E]`` done flags.
        T, E:      window shape.
        rel_off:   index where the relative-position block starts.
        pos_scale: the scale applied to positions, to recover metres.
        d_norm:    metres at which the potential decays to ~0.

    Returns:
        ``[T, E]`` float32, zero on terminal rows (no clawback across a reset).
    """
    dx = buf_s[:T, :, rel_off]
    dy = buf_s[:T, :, rel_off + 1]
    dist = np.hypot(dx, dy) / pos_scale             # back to metres
    phi = 1.0 / (1.0 + dist / d_norm)               # in (0, 1]; 1 at contact
    out = np.zeros((T, E), np.float32)
    if T > 1:
        out[:T - 1] = phi[1:T] - phi[:T - 1]
    out[buf_d[:T] > 0] = 0.0
    return out
