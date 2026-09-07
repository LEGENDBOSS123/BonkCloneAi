"""Pure learning-target builders over the [T, E] rollout buffers.

Everything here is a plain array->array function: no trainer state, no torch, no
side effects. That is deliberate — these are the parts of the update most likely
to be silently wrong (off-by-one across an episode boundary, a target that
belongs to the other seat), and pure functions can be unit-tested against a
reference in milliseconds instead of inferred from a multi-hour run.

Shape convention throughout: `[T, E]` (decision row, env), matching
rollout.RolloutBuffer. Flattening for the feedforward update is the CALLER's
job, so the recurrent path can keep the sequence axis.
"""

import numpy as np

from . import config as C


# ── advantage ───────────────────────────────────────────────────────────────
def gae_columns(r, v, d, last_v):
    """GAE(gamma, lambda) over [T, E] arrays, vectorized across the E columns.

    `last_v` [E] bootstraps a cut-off (non-terminal) tail. Returns (adv, ret).
    """
    T = r.shape[0]
    adv = np.zeros_like(r)
    gae = np.zeros(r.shape[1], dtype=np.float32)
    for t in range(T - 1, -1, -1):
        nt = 1.0 - d[t]
        next_v = last_v if t == T - 1 else v[t + 1]
        delta = r[t] + C.GAMMA * next_v * nt - v[t]
        gae = delta + C.GAMMA * C.GAE_LAMBDA * nt * gae
        adv[t] = gae
    return adv, adv + v


# ── categorical critic target ───────────────────────────────────────────────
def outcome_targets(dones, outcome, tail_dist):
    """[T*E, n_atoms] target distribution over the terminal outcome.

    With gamma = 1 and terminal-only rewards the distributional Bellman backup
    is Z(s_t) = Z(s_{t+1}) away from terminal, so the target simply propagates
    BACKWARD from each episode end. Rows whose episode has not resolved inside
    the window inherit `tail_dist` [E, n_atoms], the critic's own (detached)
    prediction at the pending state -- which is what keeps this compatible with
    bootstrapped windows instead of forcing the trainer to wait for whole
    episodes, as ppo4 had to.

    The realised class comes from `outcome` [T, E] (0=win / 1=draw / 2=loss,
    -1 = non-terminal), the ACTUAL result -- NOT nearest-atom-to-reward. The old
    shortcut merged draw and loss whenever their rewards were equal (they are,
    both -1), so the critic never learned "loss" and read every lost position as
    a draw. Pass the win/loss-swapped array to build the OPPONENT seat's target,
    or its rows would carry the learner's result.
    """
    T, E = dones.shape
    n = len(C.CRITIC_ATOMS)
    tgt = np.zeros((T, E, n), dtype=np.float32)
    cur = tail_dist.astype(np.float32).copy()
    for t in range(T - 1, -1, -1):
        d = dones[t] > 0
        if d.any():
            cls = outcome[t][d]
            oh = np.zeros((int(d.sum()), n), dtype=np.float32)
            oh[np.arange(len(cls)), cls] = 1.0
            cur = cur.copy()
            cur[d] = oh
        tgt[t] = cur
    return tgt.reshape(T * E, n)


def swap_win_loss(outcome):
    """Seat-1's outcome class array from seat-0's: win (0) <-> loss (2); draw
    (1) and non-terminal (-1) unchanged. A learner win is an opponent loss."""
    return np.where(outcome == 0, 2, np.where(outcome == 2, 0, outcome))


# ── HCA (hca.py) episode labels ─────────────────────────────────────────────
def hca_episode_labels(dones, outcome):
    """(z[T,E] int64, valid[T,E] bool) — the REALISED outcome class for every
    row of the episode it belongs to, for whichever episodes actually FINISH
    inside this window.

    Unlike `outcome_targets` (which fills an unresolved tail with the critic's
    own tail distribution, for bootstrapping), HCA needs the GROUND-TRUTH
    class only -- a row whose episode hasn't resolved by the end of the window
    gets no label at all (`valid=False`), rather than a guessed one. Backward
    propagation from each `done` row is otherwise the same idea: every row
    since the previous episode boundary shares that episode's outcome.
    """
    T, E = dones.shape
    z = np.full((T, E), -1, dtype=np.int64)
    valid = np.zeros((T, E), dtype=bool)
    cur = np.full(E, -1, dtype=np.int64)
    cur_valid = np.zeros(E, dtype=bool)
    for t in range(T - 1, -1, -1):
        d = dones[t] > 0
        if d.any():
            cur = cur.copy(); cur_valid = cur_valid.copy()
            cur[d] = outcome[t][d]
            cur_valid[d] = True
        z[t] = cur
        valid[t] = cur_valid
    return z, valid


# ── RUDDER (config.RUDDER_COEF) ──────────────────────────────────────────────
def rudder_redistribute(g, dones):
    """[T,E] redistributed reward from RUDDER's return-predictor `g` (its
    output at every row -- see `PPOAgent.rudder_head`).

    Within an episode: r[t] = g[t] - g[t-1] (telescopes back to the terminal
    reward when g is accurate). At a row that STARTS a new episode (the
    previous row was terminal): r[t] = g[t] alone, since g[-1] = 0 by
    definition for an empty prefix -- diffing against the PREVIOUS episode's
    final g would incorrectly mix two different episodes' return predictions.
    Row 0 of the WINDOW is the one case this can't resolve on its own: the
    true g one step before the window started is unknown (the rollout is a
    slice of a longer, possibly still-open episode), so it conservatively
    contributes nothing rather than risk telescoping across an untracked
    boundary. That is a small, bounded cost (one row out of T, and T is small
    -- ~20 decisions/env) traded for never being wrong about a boundary.
    """
    T, E = g.shape
    r = np.zeros((T, E), dtype=np.float32)
    if T > 1:
        r[1:] = g[1:] - g[:-1]
        new_episode = dones[:T - 1] > 0          # row t starts a new episode
        r[1:][new_episode] = g[1:][new_episode]
    return r


# ── multi-gamma auxiliary critic heads (config.AUX_GAMMA_VALUES) ────────────
def multi_gamma_targets(dones, outcome, gammas, atom_vals, tail_values=None):
    """[T,E,G] discounted-value targets, one column per gamma in `gammas`.

    Same backward-propagation shape as `outcome_targets`: walk the window from
    the end backward, and every row since the previous episode boundary shares
    that episode's outcome. The difference is each gamma DISCOUNTS the
    realised outcome value by gamma**k, where k=0 at the row that resolves the
    episode (undiscounted, matching outcome_targets exactly at gamma=1) and
    increases by 1 for every row further back before it -- the finite-horizon
    fixed point of V_gamma(s) = gamma * V_gamma(s') under a terminal-only
    reward.

    `atom_vals` orders like `outcome` (0=win/1=draw/2=loss) and should be the
    OWNING AGENT's own atom values (an exploiter's differ from the main's --
    see PPOAgent.atom_vals). Rows in a window that never resolve inherit
    `tail_values` [E,G] -- the head's own detached prediction at the pending
    state, the same bootstrap convention outcome_targets uses via tail_dist.
    """
    T, E = dones.shape
    G = len(gammas)
    gam = np.asarray(gammas, dtype=np.float32)
    atoms = np.asarray(atom_vals, dtype=np.float32)
    tgt = np.zeros((T, E, G), dtype=np.float32)
    cur = (np.zeros((E, G), dtype=np.float32) if tail_values is None
          else tail_values.astype(np.float32).copy())
    k = np.zeros(E, dtype=np.float32)     # decisions since resolution (0 at the terminal row)
    for t in range(T - 1, -1, -1):
        d = dones[t] > 0
        if d.any():
            k = k.copy(); k[d] = 0.0
            cur = cur.copy()
            cur[d] = atoms[outcome[t][d]][:, None]
        tgt[t] = cur * (gam[None, :] ** k[:, None])
        k = k + 1.0
    return tgt


# ── auxiliary critic-trunk targets ──────────────────────────────────────────
def time_targets(dones):
    """[T*E] decisions remaining until the episode resolves, normalised.

    Propagated BACKWARD from each episode end, exactly like the outcome target.
    Rows in a window that never resolves are censored: they are given the
    largest value seen so far and masked out is not possible with an MSE head,
    so instead they carry the distance to the window edge, which is a LOWER
    BOUND on the true remaining time. That biases the head toward
    under-estimating on unresolved tails -- acceptable for an auxiliary signal,
    and the alternative (dropping them) would train the head only on short
    episodes.
    """
    T, E = dones.shape
    cap = max(1.0, C.MAX_EPISODE_STEPS / C.ACTION_REPEAT)
    tgt = np.zeros((T, E), dtype=np.float32)
    since = np.zeros(E, dtype=np.float32)   # decisions until resolution
    for t in range(T - 1, -1, -1):
        since = since + 1.0
        since[dones[t] > 0] = 1.0            # resolves at this row
        tgt[t] = since
    return (tgt / cap).reshape(T * E)


def aux_targets(buf_s, buf_d, T, E):
    """(target [T*E, H, 2], mask [T*E, H]) — own displacement h steps ahead.

    obs[0:2] is the agent's own position, already POS_SCALEd, so a plain
    difference is the displacement in the same units the net sees. A horizon is
    masked off when the episode ends before it is reached: those rows would
    otherwise be trained against the NEXT episode's spawn, which is the
    stale-target bug that silently corrupted bonk4's aux data.

    `buf_s` selects the SEAT: pass the opponent's observation buffer to build
    the opponent's own-displacement target, never the learner's.
    """
    H = C.AUX_POS_HORIZONS
    tgt = np.zeros((T, E, len(H), 2), dtype=np.float32)
    msk = np.zeros((T, E, len(H)), dtype=bool)
    for hi, h in enumerate(H):
        if T - h <= 0:
            continue
        t0 = np.arange(T - h)
        tgt[t0, :, hi, :] = buf_s[t0 + h, :, 0:2] - buf_s[t0, :, 0:2]
        # valid only if no done in (t, t+h]
        crossed = np.zeros((T - h, E), dtype=bool)
        for k in range(h):
            crossed |= buf_d[t0 + k] > 0
        msk[t0, :, hi] = ~crossed
    return tgt.reshape(T * E, len(H), 2), msk.reshape(T * E, len(H))


# ── phase-1 approach shaping ────────────────────────────────────────────────
def approach_shaping(buf_s, buf_d, T, E):
    """[T,E] APPROACH reward: the rise in the bounded 1/x potential
    Phi = 1/(1 + dist/D_NORM) from this decision to the next (rel dx,dy at
    obs 24,25, POS_SCALEd; 1 at contact -> 0 far).

    Rewards the CHANGE, not the level: getting closer pays, standing still pays
    ZERO, backing off pays negative. That is the key difference from an
    occupancy reward -- with occupancy the agent just parks in a decent spot and
    collects it (measured: it stands still), because moving on fragile terrain
    risks a fatal fall. Here there is no safe standing income, so the only way to
    earn is to move toward the opponent. Zeroed at terminal rows (no clawback):
    deliberately NOT policy-invariant, so it biases the phase-1 warm-start actor
    toward approach.
    """
    rel = buf_s[:T, :, 24:26]
    dist = np.hypot(rel[:, :, 0], rel[:, :, 1]) / C.POS_SCALE
    phi = 1.0 / (1.0 + dist / C.SHAPING_D_NORM)    # bounded (0,1], closer=higher
    dd = np.zeros((T, E), dtype=np.float32)
    dd[:T - 1] = phi[1:T] - phi[:T - 1]            # positive = approached
    dd[buf_d[:T] > 0] = 0.0                        # not across resets
    return dd
