"""Recurrent actor-critic with auxiliary heads — the in-match adaptation path.

WHY: a feedforward policy on one 34-dim frame is mathematically incapable of
adapting to an opponent, because nothing about what they did 10 seconds ago
reaches it. Memory is the prerequisite for "read this player and adjust", which
no amount of extra training fixes.

    obs -> encoder MLP -> GRU -> { actor, critic, aux heads }

Unlike the feedforward agent (separate actor/critic nets), the trunk is SHARED:
two GRUs would double the dominant cost, and sharing is what lets the auxiliary
losses shape the representation the policy actually uses.

AUXILIARY HEADS — all three have free targets and all three push the hidden
state toward something specific:
  next_pos   this seat's own (dx, dy) over the next decision, in metres.
             Forces the state to encode dynamics/momentum rather than
             memorising position -> value.
  opp_action the opponent's NEXT applied action (18-way). This is the
             opponent-modelling signal: to predict what they do next, the
             hidden state has to encode who they are and what they are doing.
             It is the aux task most directly aimed at jukes/feints.
  death_soon will this seat die within DEATH_SOON_DECISIONS. Sparse terminal
             rewards give the critic almost no gradient; this densifies it
             without touching the reward function.

These are auxiliary LOSSES, not rewards. The RL objective is still
outcome-only — nothing here changes what the agent is paid for, only what its
representation is pressured to encode.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

LN_EPS = 1e-3


class RecurrentAC(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int | None = None):
        super().__init__()
        h = hidden or C.HIDDEN[-1]
        self.hidden_size = h
        self.state_dim = state_dim
        self.num_actions = num_actions

        layers, d = [], state_dim
        for w in C.HIDDEN[:-1] or [h]:
            layers += [nn.Linear(d, w, bias=False), nn.LayerNorm(w, eps=LN_EPS), nn.ReLU()]
            d = w
        self.enc = nn.Sequential(*layers)
        self.cell = nn.GRUCell(d, h)
        self.pi = nn.Linear(h, num_actions)
        self.v = nn.Linear(h, 1)
        # (dx, dy) mean metres-per-decision, one pair per C.AUX_POS_HORIZONS
        self.n_horizons = len(C.AUX_POS_HORIZONS)
        self.aux_pos = nn.Linear(h, 2 * self.n_horizons)
        self.aux_opp = nn.Linear(h, num_actions)    # opponent's next action
        self.aux_death = nn.Linear(h, 1)            # logit: dying soon

        for m in self.enc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for head in (self.pi, self.v, self.aux_pos, self.aux_opp, self.aux_death):
            nn.init.normal_(head.weight, 0.0, 0.01)
            nn.init.zeros_(head.bias)

    def initial_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def step(self, obs: torch.Tensor, h: torch.Tensor):
        """One decision for a batch of envs. Returns (logits, value, new_h)."""
        h = self.cell(self.enc(obs), h)
        return self.pi(h), self.v(h).squeeze(-1), h

    def unroll(self, obs: torch.Tensor, h0: torch.Tensor, done: torch.Tensor):
        """BPTT over a [T, B, D] sequence.

        `done[t]` marks the episode ending AT t, so the state entering t+1 is
        zeroed — without this the GRU carries memory across episode boundaries
        and learns from a discontinuity that never happens at play time.
        Looping GRUCell (rather than nn.GRU) is what makes that masking
        possible.
        """
        T = obs.shape[0]
        h = h0
        feats = []
        for t in range(T):
            if t > 0:
                h = h * (1.0 - done[t - 1]).unsqueeze(-1)
            h = self.cell(self.enc(obs[t]), h)
            feats.append(h)
        f = torch.stack(feats)                       # [T, B, H]
        return (self.pi(f), self.v(f).squeeze(-1), f, h)

    def heads_from(self, f: torch.Tensor):
        """f is [T, B, H]; the position head comes back as [T, B, horizons, 2]
        so a per-horizon validity mask can index it directly."""
        pos = self.aux_pos(f).view(*f.shape[:-1], self.n_horizons, 2)
        return pos, self.aux_opp(f), self.aux_death(f).squeeze(-1)

    # ── persistence ────────────────────────────────────────────────────────
    # Plain named tensors, NOT the tfjs MLP record format: a GRU is not an MLP
    # and must never be fed to MLP.from_records, which infers architecture from
    # kernel shapes.
    def to_records(self) -> dict:
        return {"format": "gru-v1", "hidden": self.hidden_size,
                "stateDim": self.state_dim, "numActions": self.num_actions,
                "tensors": {k: v.detach().cpu().numpy().tolist()
                            for k, v in self.state_dict().items()}}

    def load_records(self, obj: dict):
        sd = {k: torch.tensor(v) for k, v in obj["tensors"].items()}
        # The aux heads are training-only scaffolding and never ship, so a
        # checkpoint written with a different AUX_POS_HORIZONS should still
        # load its policy: drop only the tensors whose shape no longer matches
        # and let those heads start fresh. Everything else stays strict — a
        # silent shape mismatch in the GRU or policy would be a real bug.
        own = self.state_dict()
        bad = [k for k, v in sd.items()
               if k in own and v.shape != own[k].shape]
        if bad:
            print(f"note: reinitialising aux tensors with changed shapes: "
                  f"{', '.join(bad)}")
            for k in bad:
                sd[k] = own[k]
        self.load_state_dict(sd)


class RecurrentAgent:
    """Same call surface as PPOAgent where the trainer touches it."""

    def __init__(self, state_dim: int, num_actions: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.net = RecurrentAC(state_dim, num_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=C.ACTOR_LR)
        self.updates = 0
        self.stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    def initial_state(self, batch: int) -> np.ndarray:
        return np.zeros((batch, self.net.hidden_size), dtype=np.float32)

    @torch.no_grad()
    def act_batch(self, states: np.ndarray, h: np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        ht = torch.from_numpy(np.ascontiguousarray(h)).to(self.device)
        logits, v, hn = self.net.step(x, ht)
        logp_all = F.log_softmax(logits, dim=1)
        a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
        lp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        return (a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy(),
                hn.cpu().numpy())

    @torch.no_grad()
    def act_actions(self, net, states: np.ndarray, h: np.ndarray, greedy=False):
        """Actions from a frozen recurrent opponent, on that net's device."""
        dev = next(net.parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(states)).to(dev)
        ht = torch.from_numpy(np.ascontiguousarray(h)).to(dev)
        logits, _, hn = net.step(x, ht)
        a = (logits.argmax(1) if greedy
             else torch.multinomial(F.softmax(logits, 1), 1).squeeze(1))
        return a.cpu().numpy(), hn.cpu().numpy()

    # ── BPTT update ────────────────────────────────────────────────────────
    def update(self, roll: dict, entropy_coef: float = C.ENTROPY_COEF_END):
        """roll holds [T, B, ...] sequences (B = envs) plus h0 [B, H].
        Minibatching is over ENVS, never over time — splitting a sequence
        would break BPTT."""
        dev = self.device
        to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
        obs, act = to(roll["states"]), to(roll["actions"], torch.long)
        old_lp, old_v = to(roll["logps"]), to(roll["values"])
        ret, done = to(roll["returns"]), to(roll["dones"])
        h0 = to(roll["h0"])
        adv_np = roll["advantages"]
        if C.NORMALIZE_ADV:
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv = to(adv_np)
        a_pos, a_pos_m = to(roll["aux_pos"]), to(roll["aux_pos_mask"], torch.bool)
        a_opp, a_opp_m = to(roll["aux_opp"], torch.long), to(roll["aux_opp_mask"], torch.bool)
        a_die, a_die_m = to(roll["aux_death"]), to(roll["aux_death_mask"], torch.bool)

        T, B = obs.shape[0], obs.shape[1]
        # Never larger than the batch: otherwise the full-minibatch guard below
        # finds zero full minibatches and silently trains on nothing.
        seqs_per_mb = min(B, max(1, C.MINIBATCH // max(1, T)))
        stats = {k: [] for k in ("a", "c", "H", "pos", "opp", "die")}

        # Only FULL minibatches. A ragged tail is a distinct tensor shape, and
        # torch-MPS caches a compiled graph per shape without ever evicting it,
        # so the remainder would leak memory every update. The order is
        # reshuffled each epoch, so dropping the tail loses nothing
        # systematically — different envs are dropped each time.
        n_full = (B // seqs_per_mb) * seqs_per_mb
        for _ in range(C.EPOCHS):
            order = torch.randperm(B, device=dev)
            for s in range(0, max(n_full, seqs_per_mb), seqs_per_mb):
                idx = order[s:s + seqs_per_mb]
                if idx.numel() < seqs_per_mb:
                    break
                logits, v, feat, _ = self.net.unroll(obs[:, idx], h0[idx], done[:, idx])
                lp_all = F.log_softmax(logits, dim=-1)
                lp = lp_all.gather(-1, act[:, idx].unsqueeze(-1)).squeeze(-1)
                ratio = (lp - old_lp[:, idx]).exp()
                s1 = ratio * adv[:, idx]
                s2 = ratio.clamp(1 - C.CLIP_EPS, 1 + C.CLIP_EPS) * adv[:, idx]
                ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                actor_loss = -torch.min(s1, s2).mean() - entropy_coef * ent

                if C.CLIP_VALUE_LOSS:
                    vc = old_v[:, idx] + (v - old_v[:, idx]).clamp(-C.CLIP_EPS, C.CLIP_EPS)
                    critic_loss = torch.max((v - ret[:, idx]) ** 2,
                                            (vc - ret[:, idx]) ** 2).mean()
                else:
                    critic_loss = F.mse_loss(v, ret[:, idx])

                loss = actor_loss + C.VALUE_COEF * critic_loss
                p_head, o_head, d_head = self.net.heads_from(feat)

                m = a_pos_m[:, idx]
                if C.AUX_NEXT_POS_COEF > 0 and m.any():
                    l = F.mse_loss(p_head[m], a_pos[:, idx][m])
                    loss = loss + C.AUX_NEXT_POS_COEF * l
                    stats["pos"].append(l.detach())
                m = a_opp_m[:, idx]
                if C.AUX_OPP_ACTION_COEF > 0 and m.any():
                    l = F.cross_entropy(o_head[m], a_opp[:, idx][m])
                    loss = loss + C.AUX_OPP_ACTION_COEF * l
                    stats["opp"].append(l.detach())
                m = a_die_m[:, idx]
                if C.AUX_DEATH_COEF > 0 and m.any():
                    l = F.binary_cross_entropy_with_logits(d_head[m], a_die[:, idx][m])
                    loss = loss + C.AUX_DEATH_COEF * l
                    stats["die"].append(l.detach())

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), C.MAX_GRAD_NORM)
                self.opt.step()

                stats["a"].append(actor_loss.detach())
                stats["c"].append(critic_loss.detach())
                stats["H"].append(ent.detach())

        if not stats["a"]:    # nothing trained — keep the previous stats
            return self.stats
        mean = lambda k: (torch.stack(stats[k]).mean().item() if stats[k] else 0.0)
        self.updates += 1
        self.stats = {"actor_loss": mean("a"), "critic_loss": mean("c"),
                      "entropy": mean("H"), "ent_coef": entropy_coef,
                      "aux_loss": mean("pos"), "aux_opp": mean("opp"),
                      "aux_death": mean("die")}
        return self.stats

    def serialize(self) -> dict:
        return {"recurrent": self.net.to_records(), "updates": self.updates}

    def load_state(self, obj: dict):
        self.net.load_records(obj["recurrent"])
        self.updates = obj.get("updates", 0)
