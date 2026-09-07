"""Pure tensor -> tensor loss builders.

Split out of the agent so each term can be tested on hand-built tensors
without constructing a net, an optimizer or a rollout. Every function returns
an UNREDUCED per-element tensor where that makes sense; masking and
normalization belong to the caller, which is the only party that knows which
rows are free decisions and which are past burn-in.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def split_heads(logits: torch.Tensor, num_actions: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the actor's single output vector into its two heads.

    Args:
        logits:      ``[..., num_actions + n_durations]``.
        num_actions: width of the joint-action head.

    Returns:
        ``(action_logits[..., num_actions], duration_logits[..., n_durations])``.
    """
    return logits[..., :num_actions], logits[..., num_actions:]


def joint_log_prob(lpa: torch.Tensor, lpd: torch.Tensor,
                   actions: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Log-prob of the sampled (action, duration) pair.

    The two heads are conditionally independent given the state, so their
    log-probs simply add.

    Args:
        lpa:       ``[..., num_actions]`` action log-softmax.
        lpd:       ``[..., n_durations]`` duration log-softmax.
        actions:   ``[...]`` sampled action indices.
        durations: ``[...]`` sampled duration indices.

    Returns:
        ``[...]`` joint log-prob.
    """
    return (lpa.gather(-1, actions.unsqueeze(-1))
            + lpd.gather(-1, durations.unsqueeze(-1))).squeeze(-1)


def ppo_surrogate_from_ratio(ratio: torch.Tensor, advantages: torch.Tensor,
                             clip_eps: float) -> torch.Tensor:
    """Clipped surrogate objective, as a LOSS (already negated).

    Args:
        ratio:      ``[...]`` pi(a|s) / pi_old(a|s).
        advantages: ``[...]``.
        clip_eps:   trust-region half-width.

    Returns:
        ``[...]`` per-element loss; minimize.
    """
    return -torch.min(ratio * advantages,
                      ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages)


def ppo_surrogate(logp: torch.Tensor, old_logp: torch.Tensor,
                  advantages: torch.Tensor, clip_eps: float) -> torch.Tensor:
    """`ppo_surrogate_from_ratio`, computing the ratio from log-probs."""
    return ppo_surrogate_from_ratio((logp - old_logp).exp(), advantages, clip_eps)


def clip_indicators(ratio: torch.Tensor, advantages: torch.Tensor,
                    clip_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-element clip diagnostics: ``(outside_band, binding)``, both 0/1 floats.

    The two differ, and the difference matters:

    * **outside_band** — ``|ratio - 1| > clip_eps``. This is what most PPO
      implementations report as "clipfrac", so it is the number comparable to
      other codebases and to published runs.
    * **binding** — the clipped branch actually won the ``min``, so the
      surrogate's gradient for that element is ZERO. `min` selects the clipped
      term only when it is smaller, which for a positive advantage means
      ``ratio > 1 + eps`` and for a negative advantage ``ratio < 1 - eps``.
      Being outside the band on the other side leaves the gradient untouched.

    A high `outside_band` with a low `binding` means the policy is moving a lot
    but the trust region is mostly not restraining it; the two converging means
    the clip is genuinely holding updates back.
    """
    outside = ((ratio - 1.0).abs() > clip_eps).float()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    binding = ((clipped * advantages) < (ratio * advantages)).float()
    return outside, binding


def entropy_terms(lpa: torch.Tensor, lpd: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-element entropy of each head.

    Reported separately, not just summed: a collapse must be attributable to
    the ACTION head or the DURATION head specifically. Inferring it from a
    combined number is exactly what left ppo7's entropy-controller coupling
    undiagnosed for so long.

    Args:
        lpa: ``[..., num_actions]`` action log-softmax.
        lpd: ``[..., n_durations]`` duration log-softmax.

    Returns:
        ``(action_entropy[...], duration_entropy[...])``.
    """
    return (-(lpa.exp() * lpa).sum(-1), -(lpd.exp() * lpd).sum(-1))


def approx_kl_elem(logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    """Schulman's low-variance KL estimator, PER ELEMENT: ``(r - 1) - log r``.

    Unreduced so the caller can mask it. That matters here: only FREE rows
    carry a log-prob that corresponds to the action actually stored, so an
    unmasked mean over a rollout that is mostly held rows measures nothing.
    """
    log_ratio = logp - old_logp
    return (log_ratio.exp() - 1.0) - log_ratio


def approx_kl(logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    """Scalar mean of `approx_kl_elem`. Only valid when every row is free."""
    return approx_kl_elem(logp, old_logp).mean()


def categorical_critic_loss(c_out: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the outcome classifier against a SOFT target.

    Soft, not one-hot, because a bootstrapped tail carries the critic's own
    detached next-state distribution.

    Args:
        c_out:  ``[..., n_atoms]`` raw critic logits.
        target: ``[..., n_atoms]`` target distribution.

    Returns:
        ``[...]`` per-element cross-entropy.
    """
    return -(target * F.log_softmax(c_out, dim=-1)).sum(-1)


def scalar_critic_loss(v: torch.Tensor, returns: torch.Tensor,
                       old_v: torch.Tensor | None = None,
                       clip_eps: float = 0.2, clip: bool = True) -> torch.Tensor:
    """Squared value error, optionally PPO-clipped around the old value.

    Clipping matters for the phase-1 warm start, where a scalar critic faces
    high-variance terminal returns from a policy that is still moving fast.

    Args:
        v:        ``[...]`` current value prediction.
        returns:  ``[...]`` regression target.
        old_v:    ``[...]`` value at collection time; required when `clip`.
        clip_eps: clip half-width in value units.
        clip:     apply the pessimistic clipped form.

    Returns:
        ``[...]`` per-element squared error.
    """
    if clip and old_v is not None:
        v_clipped = old_v + (v - old_v).clamp(-clip_eps, clip_eps)
        return torch.max((v - returns) ** 2, (v_clipped - returns) ** 2)
    return (v - returns) ** 2


def multi_gamma_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Auxiliary multi-horizon value regression, averaged over the gammas.

    Args:
        pred:   ``[..., G]`` head output.
        target: ``[..., G]`` discounted value targets.

    Returns:
        ``[...]`` per-element mean squared error across the G horizons.
    """
    return ((pred - target) ** 2).mean(-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor,
                min_count: float = 1.0) -> torch.Tensor:
    """``sum(x * mask) / max(sum(mask), min_count)`` — a scalar.

    The floor keeps an all-masked minibatch from producing NaN.
    """
    return (x * mask).sum() / mask.sum().clamp(min=min_count)
