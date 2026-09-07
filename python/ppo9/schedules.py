"""Open-loop scalar schedules, driven by ENV STEPS.

Episodes are a poor scheduling unit here: measured episode length ranges from
47 to 1300 decisions depending on how the agent is behaving, so an
episode-based anneal moves at a rate set by the policy rather than by how much
experience has actually been collected. Steps track experience.

ppo7 also had a closed-loop SAC-style entropy controller holding combined
(action + duration) entropy at a target. It is gone: it targeted the COMBINED
entropy, so anything reshaping the duration head silently self-dampened
through it, and its interaction with the exploiter schedule was never
disentangled.
"""
from __future__ import annotations

from .config import EntropyConfig, ExploiterConfig


def entropy_coef_at(cfg: EntropyConfig, steps: int) -> float:
    """Action-head entropy coefficient after `steps` env steps.

    Linear from `coef_start` to `coef_end` over `decay_steps`, then flat.
    """
    frac = min(1.0, steps / max(1, cfg.decay_steps))
    return cfg.coef_start + (cfg.coef_end - cfg.coef_start) * frac


def duration_entropy_coef(cfg: EntropyConfig) -> float:
    """Duration-head entropy coefficient — CONSTANT, by design.

    Deliberately not derived from the action coefficient. ppo7 used a multiple
    of it (2x, later 1.5x) with a floor, and the duration head still collapsed
    whenever an exploiter drove the action coefficient toward its own floor:
    the 32- and 64-cycle buckets went to literally 0% sampled. Once a discrete
    action's probability truly reaches zero under on-policy sampling, PPO gets
    no gradient for it and it cannot come back. Decoupling removes the failure
    mode and makes the floor unnecessary.
    """
    return cfg.duration_coef


def exploiter_entropy_coef(cfg: ExploiterConfig, phase_steps: int,
                           ec_start: float, gate_hit: bool) -> float:
    """Entropy coefficient inside an exploiter phase.

    Exploiters run their own two-stage schedule instead of the main's: decay
    from this exploiter's own `ec_start` draw toward `ec_end`, then pin to
    `ec_end` once the winrate gate is hit so the sharpen tail fully commits.

    Args:
        cfg:          exploiter settings.
        phase_steps:  env steps elapsed inside this exploiter phase.
        ec_start:     this exploiter's own (possibly randomized) start value.
        gate_hit:     the graduation winrate gate has already been reached.
    """
    if gate_hit:
        return cfg.ec_end
    frac = min(1.0, phase_steps / max(1, cfg.ec_decay_steps))
    return ec_start + (cfg.ec_end - ec_start) * frac
