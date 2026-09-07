"""Completed-episode pool: the collection unit for ppo4.

The trainer used to hold a fixed [T, E] window and train on whatever resolved
inside it. That coupled three unrelated things — how much data an update sees,
how long an episode may be, and how deep the critic's BPTT runs — and it threw
away every row whose episode straddled the boundary (measured 34% at W=1000).

Here an env just accumulates its current episode. When the episode ENDS its rows
move into this pool, tagged with the outcome. An update fires once the pool
holds ROLLOUT_STEPS rows. Consequences:

* every pooled row has a true outcome, so `lab` is 100% by construction and
  there is nothing to bootstrap;
* each episode is unrolled exactly ONCE, from its own start, so h0 is always
  zero — no carried hidden state, no b_h0, no truncated-BPTT bookkeeping;
* nothing is discarded and no env ever idles waiting for a batch boundary.

The one cost is padding: sequences in a batch must be rectangular. Padding every
episode to the longest in the pool costs 2.58x on this length distribution and
would be SLOWER than the old window. Sorting into length buckets first drops
that to 1.14x at 8 buckets, which is why `critic_batches` buckets.
"""

from __future__ import annotations

import numpy as np

from . import config as C


class EpisodePool:
    """Flat row storage plus an index of (offset, length, outcome, return)."""

    def __init__(self, cap_rows: int, state_dim: int):
        self.cap = cap_rows
        self.s = np.zeros((cap_rows, state_dim), dtype=np.float32)
        self.a = np.zeros(cap_rows, dtype=np.int64)
        self.lp = np.zeros(cap_rows, dtype=np.float32)
        self.v = np.zeros(cap_rows, dtype=np.float32)
        self.oppact = np.zeros(cap_rows, dtype=np.int64)
        self.selfact = np.zeros(cap_rows, dtype=np.int64)
        self.n = 0
        self.eps: list[tuple[int, int, int, float]] = []

    def clear(self):
        self.n = 0
        self.eps.clear()

    def room(self, k: int) -> bool:
        return self.n + k <= self.cap

    def add(self, s, a, lp, v, oppact, selfact, outcome: int, ret: float):
        """Append one finished episode. Silently drops it if the pool is full —
        the caller updates as soon as the target is reached, so this only bites
        if ROLLOUT_STEPS and the capacity margin are badly mismatched."""
        k = len(a)
        if k == 0 or not self.room(k):
            return False
        o = self.n
        self.s[o:o + k] = s
        self.a[o:o + k] = a
        self.lp[o:o + k] = lp
        self.v[o:o + k] = v
        self.oppact[o:o + k] = oppact
        self.selfact[o:o + k] = selfact
        self.eps.append((o, k, outcome, ret))
        self.n += k
        return True


def _aux_pos_targets(obs, lens, horizons, pos_scale):
    """Multi-horizon own-position delta on a padded [L, n, D] batch.

    Mean metres per decision (the h-step delta / h) so every horizon sits on
    one scale. Valid only where t+h is still inside that episode — no episode
    boundary can be crossed here, because each column IS one episode.
    """
    L, n, _ = obs.shape
    H = len(horizons)
    tgt = np.zeros((L, n, H, 2), dtype=np.float32)
    msk = np.zeros((L, n, H), dtype=bool)
    rows = np.arange(L)[:, None]
    for j, h in enumerate(horizons):
        if h >= L:
            continue
        tgt[:L - h, :, j] = (obs[h:L, :, 0:2] - obs[:L - h, :, 0:2]) / pos_scale / h
        msk[:L - h, :, j] = (rows[:L - h] + h) < lens[None, :]
    return tgt, msk


def _next_action_targets(act, lens):
    """act[t+1] on a padded [L, n] batch; the episode's last row has no next."""
    L, n = act.shape
    tgt = np.zeros((L, n), dtype=np.int64)
    msk = np.zeros((L, n), dtype=bool)
    if L >= 2:
        tgt[:L - 1] = act[1:L]
        msk[:L - 1] = (np.arange(L - 1)[:, None] + 1) < lens[None, :]
    return tgt, msk


def critic_batches(pool: EpisodePool, n_buckets: int, state_dim: int):
    """Yield padded, length-bucketed batches of whole episodes.

    Each yielded column is one complete episode starting at t=0, so the caller
    unrolls from h0 = 0 with no done-mask: there is no boundary inside a
    sequence to mask. Padding past an episode's end is masked out of every loss.
    """
    if not pool.eps:
        return
    lens = np.array([e[1] for e in pool.eps], dtype=np.int64)
    order = np.argsort(lens, kind="stable")
    rew = np.asarray(C.OUTCOME_REWARD, dtype=np.float32)
    for bucket in np.array_split(order, min(n_buckets, len(order))):
        if bucket.size == 0:
            continue
        blens = lens[bucket]
        L, n = int(blens.max()), bucket.size
        obs = np.zeros((L, n, state_dim), dtype=np.float32)
        oppact = np.zeros((L, n), dtype=np.int64)
        selfact = np.zeros((L, n), dtype=np.int64)
        tgt = np.zeros((L, n, C.NUM_OUTCOMES), dtype=np.float32)
        msk = np.zeros((L, n), dtype=bool)
        for j, ei in enumerate(bucket):
            off, k, cls, _ret = pool.eps[ei]
            obs[:k, j] = pool.s[off:off + k]
            oppact[:k, j] = pool.oppact[off:off + k]
            selfact[:k, j] = pool.selfact[off:off + k]
            tgt[:k, j, cls] = 1.0        # one-hot: the episode's real outcome
            msk[:k, j] = True
        # Row index back into the flat pool, so a value computed on the padded
        # batch (e.g. the hindsight baseline) can be scattered to the rows.
        ridx = np.full((L, n), -1, dtype=np.int64)
        for j, ei in enumerate(bucket):
            off, k, _c, _r = pool.eps[ei]
            ridx[:k, j] = np.arange(off, off + k)
        pos_t, pos_m = _aux_pos_targets(obs, blens, C.AUX_POS_HORIZONS, C.POS_SCALE)
        opp_t, opp_m = _next_action_targets(oppact, blens)
        slf_t, slf_m = _next_action_targets(selfact, blens)
        yield {
            "states": obs, "targets": tgt, "target_mask": msk, "row_index": ridx,
            "aux_pos": pos_t, "aux_pos_mask": pos_m & msk[:, :, None],
            "aux_opp": opp_t, "aux_opp_mask": opp_m & msk,
            "aux_self": slf_t, "aux_self_mask": slf_m & msk,
        }


def actor_rows(pool: EpisodePool, vh: np.ndarray | None = None):
    """Flat rows plus the per-step advantage.

    RUDDER off:  A_t = R - V(prefix_t)          (Monte-Carlo residual)
    RUDDER on :  A_t = V(t) - V(t-1)            (return decomposition)

    The redistributed version telescopes to R, so it is return-equivalent: the
    same total credit, but placed on the steps where the predicted outcome
    actually moved instead of smeared over every step of the episode. The final
    step carries `R - V(last)` so the sum is exact even when the critic is
    wrong at the end.
    """
    n = pool.n
    adv = np.empty(n, dtype=np.float32)
    # CCA: swap the forward baseline for the future-conditional one. Only the
    # BASELINE changes -- the return R is untouched, so the policy gradient
    # stays unbiased provided Phi carries no own-action information.
    base = pool.v[:n]
    if vh is not None and C.HCA:
        b = C.HCA_BLEND
        base = vh if b >= 1.0 else b * vh + (1.0 - b) * pool.v[:n]
    for off, k, _cls, ret in pool.eps:
        v = base[off:off + k]
        mc = ret - v
        if not C.RUDDER:
            adv[off:off + k] = mc
            continue
        r = np.empty(k, dtype=np.float32)
        r[0] = v[0] - C.RUDDER_PRIOR
        if k > 1:
            r[1:] = v[1:] - v[:-1]
        # Return-equivalence correction: whatever the critic failed to predict
        # by the end is credited to the step that ended it.
        r[-1] += ret - v[-1]
        b = C.RUDDER_BLEND
        adv[off:off + k] = r if b >= 1.0 else b * r + (1.0 - b) * mc
    return pool.s[:n], pool.a[:n], pool.lp[:n], adv
