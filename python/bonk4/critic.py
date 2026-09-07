"""ppo4 critic: an SPPO-style outcome classifier over trajectory PREFIXES.

A GRU consumes the observation prefix o_1..o_t and predicts which of three
things the round ends in — win, loss, or draw. The scalar value is then just
the expectation under that distribution:

    V(o_1..o_t) = sum_c softmax(logits)_c * OUTCOME_REWARD[c]

and the advantage needs no bootstrapping at all:

    A_t = R_episode - V(o_1..o_t)

Why this shape:

- The target is the episode's ACTUAL outcome, a free supervised label. Unlike a
  regression critic bootstrapping off its own estimates, there is no moving
  target and no error propagation — which is the whole reason GAMMA and
  GAE_LAMBDA disappear.
- Classification over 3 classes is a much better-conditioned objective than
  regressing a 3-valued target, and it emits a calibrated p(win)/p(loss)/
  p(draw) that is directly readable.
- Prefix-conditioning is what a per-step MLP critic cannot do: whether you are
  winning depends on how the round has gone, not only on the current frame.

The hidden state IS the prefix summary because `unroll` zeroes it at episode
boundaries, so h_t always summarises o_1..o_t of the CURRENT episode and never
leaks across a reset.

Note this is not the published SPPO exactly: that predicts one sequence-level
value from the prompt, whereas here V is re-estimated at every step and
therefore sharpens as the round progresses.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import config as C

LN_EPS = 1e-3


class PrefixCritic(nn.Module):
    """Stacked GRU over the SHARED encoding z, predicting the outcome class.

    It owns no encoder: `z` arrives from the actor's trunk, so the aux heads
    and the value pressure the very features the policy acts on. Everything
    here is per-layer GRUCell rather than nn.GRU because the done-mask has to
    be applied BETWEEN timesteps — without that, h_t stops being a prefix
    summary of the current episode and leaks across resets.
    """

    def __init__(self, in_dim: int, num_actions: int,
                 hidden: int | None = None, layers: int | None = None,
                 dropout: float | None = None):
        super().__init__()
        h = hidden or C.CRITIC_HIDDEN
        n = max(1, layers if layers is not None else C.CRITIC_LAYERS)
        p = C.CRITIC_DROPOUT if dropout is None else dropout
        self.in_dim, self.num_actions = in_dim, num_actions
        self.hidden, self.layers, self.dropout_p = h, n, p
        # The trainer carries ONE flat [E, state_size] tensor, so the per-layer
        # states live concatenated. state_size is what every h0 buffer sizes to.
        self.state_size = h * n

        self.cells = nn.ModuleList(
            [nn.GRUCell(in_dim if i == 0 else h, h) for i in range(n)])
        self.drop = nn.Dropout(p) if p > 0 else nn.Identity()

        self.out = nn.Linear(h, C.NUM_OUTCOMES)
        self.n_horizons = len(C.AUX_POS_HORIZONS)
        self.aux_pos = nn.Linear(h, 2 * self.n_horizons)
        self.aux_opp = nn.Linear(h, num_actions)
        self.aux_self = nn.Linear(h, num_actions)
        for head in (self.out, self.aux_pos, self.aux_opp, self.aux_self):
            nn.init.normal_(head.weight, 0.0, 0.01)
            nn.init.zeros_(head.bias)
        # persistent=False: this is CONFIG, not a learned weight. If it went
        # into state_dict, resuming a checkpoint would restore the reward table
        # that was in force when it was saved and silently override the config
        # — so retuning DRAW_REWARD would do nothing on a resumed run.
        self.register_buffer("outcome_reward",
                             torch.tensor(C.OUTCOME_REWARD, dtype=torch.float32),
                             persistent=False)

    # ── state helpers ──────────────────────────────────────────────────────
    def initial_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.state_size, device=device)

    def _split(self, h):
        return list(h.chunk(self.layers, dim=-1)) if self.layers > 1 else [h]

    def _advance(self, z, hs):
        """One timestep through the stack. Dropout sits BETWEEN layers only —
        never on the last layer's output, which is what the heads read."""
        x, out = z, []
        for i, cell in enumerate(self.cells):
            x = cell(x, hs[i])
            out.append(x)
            if i + 1 < self.layers:
                x = self.drop(x)
        return out

    def value_of(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1) @ self.outcome_reward

    def step(self, z: torch.Tensor, h: torch.Tensor):
        hs = self._advance(z, self._split(h))
        top = hs[-1]
        logits = self.out(top)
        return logits, self.value_of(logits), torch.cat(hs, dim=-1)

    def unroll(self, z: torch.Tensor, h0: torch.Tensor, done: torch.Tensor):
        """BPTT over [T, B, in_dim] -> (logits [T,B,3], top features, final h).
        done[t] ends the episode AT t, so the state entering t+1 is zeroed."""
        T = z.shape[0]
        # FAST PATH. Since the episode-pool rewrite every sequence is a single
        # episode unrolled from h0 = 0 (see pool.critic_batches), so `done` is
        # all-zero and the per-step masking below never fires. That makes the
        # Python loop over T~300 steps pure overhead: it was ~85% of update
        # time. torch's fused GRU kernel consumes the whole sequence at once
        # and takes the SAME parameters -- GRUCell and GRU share both shapes
        # and gate order (r, z, n) -- so weights and checkpoints are untouched.
        if not bool(done.any()):
            params = []
            for c in self.cells:
                params += [c.weight_ih, c.weight_hh, c.bias_ih, c.bias_hh]
            f, hn = torch._VF.gru(
                z, torch.stack(self._split(h0)), params, True,
                len(self.cells), 0.0, self.training, False, False)
            return self.out(f), f, hn.transpose(0, 1).reshape(h0.shape)

        hs = self._split(h0)
        feats = []
        for t in range(T):
            if t > 0:
                m = (1.0 - done[t - 1]).unsqueeze(-1)
                hs = [x * m for x in hs]
            hs = self._advance(z[t], hs)
            feats.append(hs[-1])
        f = torch.stack(feats)
        return self.out(f), f, torch.cat(hs, dim=-1)

    def heads_from(self, f: torch.Tensor):
        pos = self.aux_pos(f).view(*f.shape[:-1], self.n_horizons, 2)
        return pos, self.aux_opp(f), self.aux_self(f)

    # ── persistence ────────────────────────────────────────────────────────
    def to_records(self) -> dict:
        return {"format": "prefix-critic-v2", "hidden": self.hidden,
                "layers": self.layers, "dropout": self.dropout_p,
                "inDim": self.in_dim, "numActions": self.num_actions,
                "numOutcomes": C.NUM_OUTCOMES,
                "tensors": {k: v.detach().cpu().numpy().tolist()
                            for k, v in self.state_dict().items()}}

    def load_records(self, obj: dict):
        sd = {k: torch.tensor(v) for k, v in obj["tensors"].items()}
        own = self.state_dict()
        # Drop keys this build no longer has. outcome_reward used to be a
        # PERSISTENT buffer, so every checkpoint written before that fix still
        # carries it; strict loading would reject them outright.
        extra = [k for k in sd if k not in own]
        for k in extra:
            del sd[k]
        bad = [k for k, v in sd.items() if k in own and v.shape != own[k].shape]
        miss = [k for k in own if k not in sd]
        if extra:
            print(f"note: ignoring stale critic tensors: {', '.join(extra)}")
        if bad or miss:
            print(f"note: reinitialising critic tensors "
                  f"{', '.join(bad + miss) or '(none)'}")
            for k in bad:
                sd[k] = own[k]
            for k in miss:
                sd[k] = own[k]
        self.load_state_dict(sd)


def propagate_terminal(done: np.ndarray, value: np.ndarray, T: int, E: int):
    """Same backward scan as outcome_labels, but for a FLOAT written at the
    terminal row — the episode's actual return. Used so the actor can be paid a
    time-decayed win while the critic keeps its flat 3-class target."""
    out = np.zeros((T, E), dtype=np.float32)
    cur = np.zeros(E, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        ends = done[t] > 0
        if ends.any():
            cur = np.where(ends, value[t], cur)
        out[t] = cur
    return out


def outcome_labels(done: np.ndarray, outcome: np.ndarray, T: int, E: int):
    """Label every row with the outcome of the episode it belongs to.

    Returns (labels [T,E] int64, valid [T,E] bool). Walking BACKWARDS, a
    terminal row's outcome propagates to every earlier row of that episode.

    Rows after the last terminal in the window are INVALID: their episode has
    not finished, so its outcome is genuinely unknown. A bootstrapping critic
    would estimate it; this one cannot, and inventing a label would teach the
    critic something false. Those rows are dropped from both the critic and
    the actor update — the one real cost of removing bootstrapping.
    """
    lab = np.zeros((T, E), dtype=np.int64)
    val = np.zeros((T, E), dtype=bool)
    cur = np.zeros(E, dtype=np.int64)
    seen = np.zeros(E, dtype=bool)
    for t in range(T - 1, -1, -1):
        ends = done[t] > 0
        if ends.any():
            cur = np.where(ends, outcome[t], cur)
            seen = seen | ends
        lab[t] = cur
        val[t] = seen
    return lab, val
