"""Actor with a transplanted movement SKILL (actor-v3).

WHY v2 DID NOT WORK
-------------------
v2 kept only the middle of the pre-trained net and threw away its input and
output layers:

    obs -> trunk -> proj_in -> [FROZEN CORE] -> concat with trunk -> logits

Three things go wrong with that, and all three were observed:

1. NOTHING PINS THE INPUT to the region the core understands. `proj_in` may map
   anywhere in R^64. LayerNorm fixes the per-sample mean and variance of `u`
   but not its DIRECTION, so the core is still evaluated far outside its
   training manifold, where its learned structure means nothing and it
   degenerates into an arbitrary fixed nonlinearity. Measured on the v2 actor:
   u std 0.583 against a pre-training value of 0.285 -- already 2x outside.

2. THE OUTPUT IS AN UNGROUNDED CODE. The core's 64-d output only meant
   something in combination with the discarded `l_out`. Stripped of it, the
   actor has to learn from scratch what those 64 numbers imply about which
   button to press -- the same difficulty as learning to move in the first
   place. The transplant buys nothing.

3. THE SKIP PATH DOMINATES. `fuse` sees concat[h_task(256), m(64)] and h_task
   alone is already sufficient to pick an action. Gradient descent prefers the
   directly-trainable path, so the core branch decays to dead weight -- and
   nothing in the architecture reveals that it has.

WHAT v3 DOES INSTEAD
--------------------
Keep the WHOLE pre-trained skill (l_in + core + l_out) frozen, and drive it
through the interface it was actually trained on:

    obs[34] ─┬─► trunk 34→256→256 ─┬─► goal_head 256→4 ─► bounded Δ
             │                     │            │
             │                     │      g = own_pos + Δ   (absolute, scaled)
             │                     │            │
             │        own = obs[0:12] ──────────┴──► [FROZEN l_in→core→l_out]
             │                     │                            │
             │                     │                     logits_move (18)
             │                     │                            │
             └─────────────────────┴──► out 256→18 ──(+)── prior_scale · ┘
                                                        │
                                                      logits

Each failure mode above is answered structurally:

1. IN-DISTRIBUTION BY CONSTRUCTION. The goal is emitted as a bounded offset
   from the agent's CURRENT position -- tanh-squashed to exactly the radius the
   pre-trainer sampled goals in. The frozen stack is therefore never asked
   about a (state, goal) pair outside what it was trained on. There is no way
   for the actor to walk it off the manifold.

2. BOTH ENDS ARE GROUNDED. The input is (own_state, goal) in physical units and
   the output is action logits. The actor never has to invent or decode an
   arbitrary embedding: it chooses WHERE TO GO (4 numbers with meaning) and
   reads back WHICH BUTTON -- both already mean what they say.

3. RESIDUAL, NOT CONCATENATED. The skill contributes in ACTION space and `out`
   is initialised near zero, so at step 0 the policy *is* the movement policy
   -- the agent can already move competently before learning anything. `out`
   then learns a correction on top. This is Residual Policy Learning
   (Johannink et al. 2018; Silver et al. 2018). The skill cannot be quietly
   ignored: its weight is one logged scalar, `prior_scale`, so if training
   drives it to zero we can SEE that it did.

What the actor learns is exactly what the original idea intended -- how to USE
the skill -- but over a grounded interface instead of a free-form code, which
is what makes it learnable rather than merely well-posed on paper.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from . import config as C
from .movement import MovementCore

LN_EPS = 1e-3


class MovementSkill(nn.Module):
    """The frozen pre-trained stack: (own_state, goal) -> action logits."""

    def __init__(self, own_dim: int, goal_dim: int, d_emb: int, hidden: int,
                 num_actions: int):
        super().__init__()
        self.own_dim, self.goal_dim, self.d_emb = own_dim, goal_dim, d_emb
        self.l_in = nn.Sequential(
            nn.Linear(own_dim + goal_dim, d_emb),
            nn.LayerNorm(d_emb, eps=LN_EPS), nn.ReLU())
        # The REAL MovementCore class, not a copy of its layer stack: the
        # pre-trainer saves it as `net.0.weight` etc., and any local
        # re-declaration silently diverges from it on key names or structure.
        self.core = MovementCore(d_emb, hidden)
        self.l_out = nn.Linear(d_emb, num_actions)

    def forward(self, own, goal):
        return self.l_out(self.core(self.l_in(torch.cat([own, goal], dim=-1))))


class ActorNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden: list[int] | None = None, d_emb: int | None = None):
        super().__init__()
        hid = list(hidden or C.HIDDEN)
        d = d_emb or C.MOVE_EMB
        self.state_dim, self.num_actions = state_dim, num_actions
        self.hidden, self.d_emb = hid, d

        layers, prev = [], state_dim
        for w in hid:
            layers += [nn.Linear(prev, w, bias=False),
                       nn.LayerNorm(w, eps=LN_EPS), nn.ReLU()]
            prev = w
        self.trunk = nn.Sequential(*layers)
        self.z_dim = prev

        self.use_skill = bool(C.MOVE_TRANSPLANT)
        # A BUFFER, not a bool: _freeze() and load_state_dict() copy tensors
        # only, so a plain attribute would leave every frozen snapshot running
        # without the prior while the live agent used it -- the pool and the
        # learner would be playing different policies.
        self.register_buffer("skill_flag", torch.zeros(()))
        if self.use_skill:
            self.own_dim, self.goal_dim = C.MOVE_OWN_DIM, C.MOVE_GOAL_DIM
            self.skill = MovementSkill(self.own_dim, self.goal_dim, d,
                                       C.MOVE_HIDDEN, num_actions)
            for p in self.skill.parameters():
                p.requires_grad_(False)
            self.goal_head = nn.Linear(prev, self.goal_dim)
            # Xavier, NOT a near-zero init. A near-zero goal head emits offset
            # ~0, i.e. it asks the skill to STAND STILL -- the exact degenerate
            # behaviour the transplant exists to prevent, and measured at 0.6 m
            # of requested travel against an 8 m bound. The pre-trainer sampled
            # goals uniformly in +-r, so a typical |offset| is ~r/2; a spread
            # this size makes the agent move toward VARIED goals from step 0,
            # which is where the exploration benefit was supposed to come from.
            nn.init.xavier_uniform_(self.goal_head.weight)
            nn.init.zeros_(self.goal_head.bias)
            # How strongly the movement prior speaks. LEARNED, and logged --
            # this one scalar is what makes "the transplant is being ignored"
            # an observable event instead of a silent one.
            self.prior_scale = nn.Parameter(torch.ones(()))
            # Goal bounds in the scaled units the skill was trained on; filled
            # from the checkpoint by load_skill().
            self.register_buffer("goal_bound", torch.ones(self.goal_dim))

        self.out = nn.Linear(prev, num_actions)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        # Near-zero so the policy STARTS as the movement prior and learns a
        # correction, instead of starting random and having to discover that
        # the prior was worth listening to.
        nn.init.normal_(self.out.weight, 0.0, 0.01)
        nn.init.zeros_(self.out.bias)

    # ── forward ────────────────────────────────────────────────────────────
    def features(self, x):
        """The representation the critic and the aux heads read."""
        return self.trunk(x)

    def goal_of(self, x, h=None):
        """-> the ABSOLUTE goal (scaled units) this state maps to.

        Bounded as an offset from the agent's own position, so it always lands
        inside the region the pre-trainer sampled goals from.
        """
        h = self.trunk(x) if h is None else h
        d = torch.tanh(self.goal_head(h)) * self.goal_bound
        own_pos = x[..., :2]
        # Position is an OFFSET from where we are; velocity is an absolute
        # target -- matching how the pre-trainer sampled each of them.
        return torch.cat([own_pos + d[..., :2], d[..., 2:]], dim=-1)

    def move_logits(self, x, h=None):
        own = x[..., :self.own_dim]
        lm = self.skill(own, self.goal_of(x, h))
        lm = lm - lm.mean(dim=-1, keepdim=True)
        # STANDARDISED, not just centred. The skill was pre-trained down to
        # entropy 0.5, so its logits are already extremely peaked and centring
        # does nothing to their magnitude. Once prior_scale became trainable it
        # grew 1.0 -> 2.0, the combined logits saturated the softmax (H fell to
        # 0.21), the resulting tiny pi(a|s) blew the importance ratios up
        # (kl 0.03 target -> 3223) and training NaN'd. Unit scale makes the
        # prior's contribution depend on prior_scale alone.
        return lm / (lm.std(dim=-1, keepdim=True) + 1e-6)

    @property
    def skill_ready(self) -> bool:
        return bool(self.skill_flag.item())

    def logits_from(self, x, h):
        """Policy logits given PRE-COMPUTED trunk features.

        Exists so callers that already hold `features(x)` (the shared-trunk
        path, which needs them for the value and aux heads) can still get the
        FULL policy without a second trunk pass. Routing those callers through
        `out(h)` alone silently drops the movement prior, which makes the
        acting policy and the updated policy two different distributions.
        """
        logits = self.out(h)
        if self.use_skill and self.skill_ready:
            # Clamped: an unbounded weight on a sharp prior is what diverged.
            # clamp keeps gradient inside the range and kills it outside.
            ps = self.prior_scale.clamp(0.0, C.MOVE_PRIOR_MAX)
            logits = logits + ps * self.move_logits(x, h)
        return logits

    def forward(self, x):
        return self.logits_from(x, self.trunk(x))

    # ── monitoring ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def skill_stats(self, x) -> dict:
        """Is the transplant alive, and is it being asked sensible questions?"""
        if not (self.use_skill and self.skill_ready):
            return {}
        h = self.trunk(x)
        g = self.goal_of(x, h)
        dist = (g[..., :2] - x[..., :2]).norm(dim=-1)
        lm = (self.prior_scale.clamp(0.0, C.MOVE_PRIOR_MAX)
              * self.move_logits(x, h)).abs().mean()
        lt = self.out(h).abs().mean()
        return {
            "prior": float(self.prior_scale),
            # Mean |goal - own_pos| in METRES. If this collapses toward 0 the
            # actor is asking the skill to stand still -- the degenerate case,
            # and exactly the behaviour this whole module exists to prevent.
            "goal_m": float(dist.mean()) / C.POS_SCALE,
            # Share of the final logit magnitude contributed by the prior.
            "prior_frac": float(lm / (lm + lt + 1e-8)),
        }

    # ── pre-trained skill ──────────────────────────────────────────────────
    def load_skill(self, path: str) -> bool:
        if not self.use_skill:
            return False
        with open(path) as f:
            obj = json.load(f)
        if obj.get("format") != "move-core-v2":
            print(f"  WARNING: {path} is {obj.get('format')!r}, not "
                  f"'move-core-v2' (no l_in/l_out). Re-run pretrain_move; the "
                  f"movement prior is DISABLED for this run.")
            return False
        if obj.get("d_emb") != self.d_emb:
            print(f"  WARNING: skill d_emb {obj.get('d_emb')} != {self.d_emb}; "
                  f"movement prior DISABLED")
            return False
        t = lambda dd: {k: torch.tensor(v) for k, v in dd.items()}   # noqa: E731
        self.skill.l_in.load_state_dict(t(obj["l_in"]))
        self.skill.core.load_state_dict(t(obj["core"]))
        self.skill.l_out.load_state_dict(t(obj["l_out"]))
        with torch.no_grad():
            self.goal_bound.copy_(torch.tensor(
                [obj["goal_radius"], obj["goal_radius"],
                 obj["goal_vx"], obj["goal_vy"]], dtype=torch.float32))
        for p in self.skill.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            self.skill_flag.fill_(1.0)
        return True

    # ── persistence ────────────────────────────────────────────────────────
    def to_records(self) -> dict:
        return {"format": "actor-v3", "stateDim": self.state_dim,
                "numActions": self.num_actions, "hidden": self.hidden,
                "dEmb": self.d_emb, "useSkill": self.use_skill,
                "skillReady": self.skill_ready,
                "tensors": {k: v.detach().cpu().numpy().tolist()
                            for k, v in self.state_dict().items()}}

    def load_records(self, obj):
        if not isinstance(obj, dict) or obj.get("format") != "actor-v3":
            got = obj.get("format") if isinstance(obj, dict) else type(obj)
            raise ValueError(f"not an actor-v3 record (got {got!r}); older "
                             f"transplant checkpoints are not loadable")
        sd = {k: torch.tensor(v) for k, v in obj["tensors"].items()}
        own = self.state_dict()
        drop = [k for k in sd if k not in own or sd[k].shape != own[k].shape]
        for k in drop:
            del sd[k]
        if drop:
            print(f"note: actor reinitialising {', '.join(drop)}")
            for k in drop:
                sd[k] = own[k]
        self.load_state_dict(sd)
        # The frozen skill travels WITH the checkpoint, so a restored actor is
        # self-contained: pool snapshots and deploy never re-read the JSON.
        if self.use_skill:
            for p in self.skill.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_records(cls, obj):
        net = cls(obj["stateDim"], obj["numActions"], obj["hidden"], obj["dEmb"])
        net.load_records(obj)
        return net
