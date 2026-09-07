"""PPOAgent: clipped-surrogate PPO over the joint (action, duration) policy.

Two heads are packed into ONE actor output vector — ``[0:num_actions]`` are the
joint-action logits, the rest are the FiGAR duration logits. One net keeps the
records format and the browser deploy simple; the split happens wherever the
policy is read. The heads are conditionally independent given the state, so
their log-probs and entropies simply add.

The critic is a 3-way classifier over {win, draw, loss} rather than a scalar
regressor. That is only sound because `gamma = 1` and the reward is
terminal-only, which makes the distributional Bellman backup a plain backward
propagation of the realized outcome.

There is exactly ONE update path: truncated BPTT over `[T, E]` sequences.
ppo7's second (flat-row, feedforward) path is gone with the conv encoder, and
with it the position/time auxiliary heads and the dead advantage-shaping term
that lived only there.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import losses
from .config import CriticConfig, EntropyConfig, FigarConfig, NetConfig, PpoConfig
from .mingru import MinGRUNet, Records
from .nets import build_actor, build_critic

Buf = dict[str, np.ndarray]


class PPOAgent:
    """One trainable policy/value pair, plus its auxiliary heads."""

    def __init__(self, state_dim: int, num_actions: int, *,
                 net: NetConfig, ppo: PpoConfig, critic: CriticConfig,
                 figar: FigarConfig, entropy: EntropyConfig,
                 atoms: tuple[float, ...], device: str | torch.device = "cpu",
                 is_exploiter: bool = False, is_staller: bool = False,
                 clip_eps: float | None = None, actor_lr: float | None = None,
                 epochs: int | None = None) -> None:
        """
        Args:
            state_dim:   net input width (env obs + the FiGAR hold feature).
            num_actions: joint-action head width.
            atoms:       outcome values ordered (win, draw, loss). An EXPLOITER
                         passes its own — a draw is worth something different
                         to it — while sharing all other machinery.
            is_exploiter/is_staller: informational; the league reads them.
            clip_eps/actor_lr/epochs: per-agent overrides. Exploiters get a
                         looser trust region and bigger steps because they must
                         DIVERGE from the frozen main to find a hole; a
                         conservative clip just keeps them hugging it.
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.cfg_net, self.cfg_ppo, self.cfg_critic = net, ppo, critic
        self.cfg_figar, self.cfg_entropy = figar, entropy
        self.is_exploiter, self.is_staller = is_exploiter, is_staller

        self.clip_eps = ppo.clip_eps if clip_eps is None else clip_eps
        self.actor_lr = ppo.actor_lr if actor_lr is None else actor_lr
        self.epochs = ppo.epochs if epochs is None else epochs

        self.n_dur = figar.n
        self.atom_vals = tuple(atoms)
        self.atoms = torch.tensor(self.atom_vals, dtype=torch.float32,
                                  device=self.device)
        self.cat = bool(critic.categorical)

        # A tiny shim so the net builders can stay config-driven without the
        # agent needing the whole root config.
        class _Shim:
            def __init__(s, agent: "PPOAgent") -> None:
                s.net, s.critic, s.figar = agent.cfg_net, agent.cfg_critic, agent.cfg_figar
                s.agent_state_dim = agent.state_dim
                s.actor_out_dim = agent.num_actions + agent.n_dur

        shim = _Shim(self)
        self.actor: MinGRUNet = build_actor(shim, self.device)     # type: ignore[arg-type]
        self.critic: MinGRUNet = build_critic(shim, len(self.atom_vals),
                                              self.device)         # type: ignore[arg-type]

        # Auxiliary multi-horizon value heads, riding the CRITIC's own trunk —
        # zero extra forward pass (see MinGRUNet.forward_seq's `trunk`). They
        # never touch the advantage or the policy gradient.
        self.n_gammas = len(critic.aux_gamma_values)
        self.gamma_head = torch.nn.Linear(net.hidden[-1], self.n_gammas).to(self.device)
        torch.nn.init.zeros_(self.gamma_head.bias)
        torch.nn.init.normal_(self.gamma_head.weight, 0.0, 0.01)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # gamma_head rides with the critic: same trunk, same learning rate, and
        # crucially NOT in the actor's optimizer.
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.gamma_head.parameters()),
            lr=ppo.critic_lr)

        self._dur_cycles = torch.tensor(figar.durations, dtype=torch.float32,
                                        device=self.device)
        self.updates = 0
        self.stats: dict[str, float] = {}

    # ── acting ─────────────────────────────────────────────────────────────
    def value_of(self, out: torch.Tensor) -> torch.Tensor:
        """Critic output -> scalar value.

        Categorical: a convex combination of the atoms, hence STRUCTURALLY
        bounded in ``[min(atoms), max(atoms)]`` and unable to diverge.
        """
        if not self.cat:
            return out.squeeze(-1)
        return torch.softmax(out, dim=-1) @ self.atoms

    @torch.no_grad()
    def act_step(self, states: np.ndarray, ha: np.ndarray, hc: np.ndarray
                 ) -> tuple[np.ndarray, ...]:
        """One recurrent collection step for the learner seat.

        Args:
            states: ``[n, state_dim]`` observations.
            ha, hc: ``[n, H]`` previous actor / critic hidden.

        Returns:
            ``(actions[n], durations[n], logps[n], values[n], ha2[n,H], hc2[n,H])``.
        """
        x = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        hat = torch.from_numpy(np.ascontiguousarray(ha)).to(self.device)
        hct = torch.from_numpy(np.ascontiguousarray(hc)).to(self.device)

        logits, ha2 = self.actor.step(x, hat)
        la, ld = losses.split_heads(logits, self.num_actions)
        lpa, lpd = F.log_softmax(la, -1), F.log_softmax(ld, -1)
        a = torch.multinomial(lpa.exp(), 1).squeeze(1)
        d = torch.multinomial(lpd.exp(), 1).squeeze(1)
        logp = losses.joint_log_prob(lpa, lpd, a, d)

        c_out, hc2 = self.critic.step(x, hct)
        v = self.value_of(c_out)
        return (a.cpu().numpy(), d.cpu().numpy(), logp.cpu().numpy(),
                v.cpu().numpy(), ha2.cpu().numpy(), hc2.cpu().numpy())

    @torch.no_grad()
    def act_actions_step(self, net: MinGRUNet, states: np.ndarray,
                         h: np.ndarray, greedy: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A frozen opponent net acts, carrying its own hidden state.

        Runs on the NET's device: pool nets stay on CPU even when the learner
        trains on GPU, because many small batches lose to dispatch overhead.

        Args:
            greedy: ``[n]`` bool, or None for all-sampled. Where set, BOTH
                heads take the argmax instead of sampling. Only ever applies to
                the OPPONENT seat — the learner must keep sampling or its
                stored log-probs stop being on-policy and PPO's ratio is
                meaningless. See `LeagueConfig.opp_greedy_prob`.

        Returns:
            ``(actions[n], durations[n], h2[n, H])``.
        """
        dev = next(net.parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(states)).to(dev)
        ht = torch.from_numpy(np.ascontiguousarray(h)).to(dev)
        logits, h2 = net.step(x, ht)
        la, ld = losses.split_heads(logits, self.num_actions)
        a = torch.multinomial(F.softmax(la, -1), 1).squeeze(1)
        d = torch.multinomial(F.softmax(ld, -1), 1).squeeze(1)
        if greedy is not None and greedy.any():
            g = torch.from_numpy(np.ascontiguousarray(greedy)).to(dev)
            a = torch.where(g, la.argmax(-1), a)
            d = torch.where(g, ld.argmax(-1), d)
        return a.cpu().numpy(), d.cpu().numpy(), h2.cpu().numpy()

    # ── update ─────────────────────────────────────────────────────────────
    def update(self, buf: Buf, entropy_coef: float,
               duration_entropy_coef: float,
               freeze_actor: bool | None = None) -> dict[str, float]:
        """Truncated-BPTT PPO update over the learner's ``[T, E]`` rollout.

        The recurrent hidden is carried ACROSS rollouts during collection
        (reset only at episode ends), so each rollout is a contiguous chunk
        re-forwarded from its stored start hidden. Minibatches are over
        ENVIRONMENTS: each is a full length-T sequence.

        Args:
            buf: numpy arrays —
                ``states[T,E,D]``, ``actions[T,E]``, ``durations[T,E]``,
                ``free[T,E]``, ``old_logp[T,E]``, ``returns[T,E]``,
                ``advantages[T,E]``, ``values[T,E]``, ``dones[T,E]``,
                ``ha0[E,H]``, ``hc0[E,H]``, and optionally
                ``outcome_target[T,E,n_atoms]`` (required when categorical) and
                ``gamma_target[T,E,G]``.
            entropy_coef:          action-head entropy weight.
            duration_entropy_coef: duration-head entropy weight (constant).
            freeze_actor: overrides `ppo.freeze_actor` for THIS call only, so a
                caller can hold the actor still for a bounded warm-up without
                mutating the stored config (see
                `ExploiterConfig.actor_frozen`). None keeps the config's value.

        Returns:
            Scalar diagnostics for the log line.
        """
        dev = self.device
        cfgp, cfgc = self.cfg_ppo, self.cfg_critic

        def to(a: np.ndarray, dt: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.as_tensor(a, dtype=dt, device=dev)

        T, E, _ = buf["states"].shape
        S = to(buf["states"])
        A = to(buf["actions"], torch.long)
        Du = to(buf["durations"], torch.long)
        FR = to(buf["free"])
        OLP = to(buf["old_logp"])
        RET = to(buf["returns"])
        OLDV = to(buf["values"])
        DON = to(buf["dones"])
        OC = to(buf["outcome_target"]) if self.cat else None
        GT = to(buf["gamma_target"]) if buf.get("gamma_target") is not None else None
        ha0, hc0 = to(buf["ha0"]), to(buf["hc0"])

        adv = buf["advantages"].astype(np.float32)
        if cfgp.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ADV = to(adv)

        # Env-major sequences [E, T, *]. `reset` zeros the hidden at each
        # new-episode row (the row AFTER a done), matching collection.
        St = S.transpose(0, 1).contiguous()
        reset = torch.zeros(E, T, device=dev)
        reset[:, 1:] = DON.transpose(0, 1)[:, :-1]
        # Burn-in rows only warm the recomputed hidden; they carry no loss.
        loss_mask = torch.zeros(T, device=dev)
        loss_mask[cfgp.burn_in:] = 1.0

        idx_env = np.arange(E)
        mb_size = max(1, cfgp.minibatch // max(1, T))
        a_losses, c_losses, kls = [], [], []
        ent_all, ent_a_all, ent_d_all, holds, gamma_losses = [], [], [], [], []
        logit_sq: list[float] = []
        clip_out, clip_bind = [], []
        n_mb = n_kl_stopped = 0

        for _ in range(self.epochs):
            np.random.shuffle(idx_env)
            for s in range(0, E, mb_size):
                mb = idx_env[s:s + mb_size]
                xb = St[mb]                                   # [B, T, D]
                logits, _, _ = self.actor.forward_seq(xb, ha0[mb], reset[mb])
                c_out, _, c_trunk = self.critic.forward_seq(xb, hc0[mb], reset[mb])

                la, ld = losses.split_heads(logits, self.num_actions)
                lpa, lpd = F.log_softmax(la, -1), F.log_softmax(ld, -1)
                a_mb = A[:, mb].transpose(0, 1)
                d_mb = Du[:, mb].transpose(0, 1)
                logp = losses.joint_log_prob(lpa, lpd, a_mb, d_mb)
                old = OLP[:, mb].transpose(0, 1)
                advb = ADV[:, mb].transpose(0, 1)

                ratio = (logp - old).exp()
                surr = losses.ppo_surrogate_from_ratio(ratio, advb, self.clip_eps)
                ent_a, ent_d = losses.entropy_terms(lpa, lpd)

                # The actor trains on FREE rows past burn-in only.
                fm = FR[:, mb].transpose(0, 1) * loss_mask[None, :]
                denom = fm.sum().clamp(min=1.0)
                ent_bonus = (entropy_coef * (ent_a * fm).sum()
                             + duration_entropy_coef * (ent_d * fm).sum())
                actor_loss = ((surr * fm).sum() - ent_bonus) / denom
                if cfgp.logit_l2 > 0.0:
                    # sum over classes, averaged over the rows the actor
                    # trains on — the same `fm` mask, so this regularises
                    # exactly where the policy gradient acts and is inert
                    # whenever the actor is frozen (nothing backwards it).
                    # Covers BOTH heads: the duration head is the one measured
                    # collapsing, and its logits saturate the same way.
                    l2 = (logits.pow(2).sum(-1) * fm).sum() / denom
                    actor_loss = actor_loss + cfgp.logit_l2 * l2
                    logit_sq.append(float(l2.detach()))

                # The critic trains on EVERY row past burn-in.
                cm = loss_mask[None, :].expand(len(mb), T)
                cden = cm.sum().clamp(min=1.0)
                if self.cat:
                    closs = losses.categorical_critic_loss(
                        c_out, OC[:, mb].transpose(0, 1))
                else:
                    closs = losses.scalar_critic_loss(
                        self.value_of(c_out), RET[:, mb].transpose(0, 1),
                        OLDV[:, mb].transpose(0, 1), self.clip_eps,
                        cfgc.clip_value_loss)
                critic_loss = cfgc.value_coef * (closs * cm).sum() / cden

                if GT is not None and cfgc.aux_gamma_coef > 0.0:
                    # Every row has a legitimate (possibly bootstrapped) target,
                    # so the burn-in mask is the right one here.
                    gl = losses.multi_gamma_loss(self.gamma_head(c_trunk),
                                                 GT[:, mb].transpose(0, 1))
                    gamma_loss = (gl * cm).sum() / cden
                    critic_loss = critic_loss + cfgc.aux_gamma_coef * gamma_loss
                    gamma_losses.append(float(gamma_loss.detach()))

                # Both forwards ran BEFORE either loss is used, and the two loss
                # terms touch DISJOINT parameter sets (the actor loss has no
                # autograd path to critic params or vice versa) — so one
                # combined backward is exactly equivalent to two separate calls,
                # with one graph traversal instead of two. This halves backward
                # calls on the hottest loop in the trainer.
                # Per-minibatch trust-region stop. Measured on FREE rows only,
                # for the same reason the kl STAT is: a held row's stored
                # log-prob belongs to an action that was discarded.
                mb_kl = float(losses.masked_mean(
                    losses.approx_kl_elem(logp.detach(), old), fm))
                n_mb += 1
                kl_stopped = cfgp.kl_stop > 0.0 and mb_kl > cfgp.kl_stop
                n_kl_stopped += int(kl_stopped)

                frozen = (cfgp.freeze_actor if freeze_actor is None
                          else bool(freeze_actor))
                do_actor = not frozen and bool(denom > 0) and not kl_stopped
                self.critic_opt.zero_grad(set_to_none=True)
                if do_actor:
                    self.actor_opt.zero_grad(set_to_none=True)
                    (actor_loss + critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                   cfgp.max_grad_norm)
                    self.actor_opt.step()
                else:
                    critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                               cfgp.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    a_losses.append(float(actor_loss.detach()))
                    c_losses.append(float(critic_loss.detach()))
                    ent_all.append(float(((ent_a + ent_d) * fm).sum() / denom))
                    ent_a_all.append(float((ent_a * fm).sum() / denom))
                    ent_d_all.append(float((ent_d * fm).sum() / denom))
                    # Masked to FREE rows, like every other stat here. On a
                    # HELD row `old_logp` is the log-prob of a sampled action
                    # that was then DISCARDED in favour of the latched one, so
                    # the ratio there is meaningless — and at a typical
                    # decided% of 3-40% an unmasked mean is mostly that
                    # garbage, reporting a trust-region violation that is not
                    # happening.
                    kls.append(mb_kl)
                    holds.append(float((self._dur_cycles[d_mb] * fm).sum() / denom))
                    # Clip diagnostics, masked to free rows like everything else.
                    out_i, bind_i = losses.clip_indicators(ratio, advb, self.clip_eps)
                    clip_out.append(float(losses.masked_mean(out_i, fm)))
                    clip_bind.append(float(losses.masked_mean(bind_i, fm)))

        self.updates += 1
        mean = lambda xs: sum(xs) / len(xs)                   # noqa: E731
        self.stats = {
            "actor_loss": mean(a_losses),
            "critic_loss": mean(c_losses),
            "entropy": mean(ent_all),
            # Split out of the combined H so a collapse can be pinned to the
            # ACTION head specifically rather than inferred.
            "action_entropy": mean(ent_a_all),
            "duration_entropy": mean(ent_d_all),
            "ent_coef": entropy_coef,
            "kl": mean(kls),
            # Fraction of free rows whose ratio left the trust region, and the
            # fraction where the clip actually zeroed the gradient. See
            # losses.clip_indicators for why those are not the same number.
            "clip_frac": mean(clip_out),
            "clip_binding": mean(clip_bind),
            "mean_hold": mean(holds) if holds else 1.0,
        }
        if gamma_losses:
            self.stats["gamma_loss"] = mean(gamma_losses)
        if cfgp.kl_stop > 0.0:
            self.stats["kl_stopped_pct"] = 100.0 * n_kl_stopped / max(1, n_mb)
        if logit_sq:
            # Reported as RMS per logit so it is directly comparable to the
            # `ppo9.diagnosis` figure and to the ~3 healthy reference.
            n_logits = self.num_actions + self.n_dur
            self.stats["logit_rms"] = float(np.sqrt(mean(logit_sq) / n_logits))
        return self.stats

    # ── persistence ────────────────────────────────────────────────────────
    def serialize(self) -> dict[str, Any]:
        """Actor + critic records. Auxiliary heads are NOT saved: they are
        representation scaffolding, cheap to re-learn, and keeping them out
        means a checkpoint stays loadable when they change."""
        return {"actor": self.actor.to_records(),
                "critic": self.critic.to_records(),
                "updates": self.updates}

    def load_state(self, obj: dict[str, Any]) -> None:
        self.actor.load_records(obj["actor"])
        self.critic.load_records(obj["critic"])
        self.updates = int(obj.get("updates", 0))

    def actor_records(self) -> Records:
        return self.actor.to_records()
