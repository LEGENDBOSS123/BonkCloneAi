"""Scalar hyper-parameter schedules driven by ENV STEPS.

Steps, not episodes: episode length varies by more than an order of magnitude
over training, so an episode-based anneal moves at a rate that depends on how
the agent happens to be playing rather than on how much experience it has seen.

`ppo.PPOAgent.adapt_entropy` is the *other* half of the entropy story (a
closed-loop controller used in the league phase); these are the open-loop
schedules used everywhere else.
"""

from . import config as C


def entropy_coef_at(steps: int) -> float:
    """Action-head entropy coef: linear anneal START -> END over
    ENTROPY_DECAY_STEPS, then flat."""
    frac = min(1.0, steps / C.ENTROPY_DECAY_STEPS)
    return C.ENTROPY_COEF_START + (C.ENTROPY_COEF_END - C.ENTROPY_COEF_START) * frac


def duration_entropy_coef_at(steps: int) -> float:
    """Separate anneal for the FiGAR duration head (see config.DURATION_ENTROPY_*).

    The duration head collapses under the action head's coef, so it gets its own,
    stronger schedule to keep long holds sampled and exploration alive.

    NOTE: currently unused by the trainer, which derives the duration coef as
    config.DURATION_ENTROPY_MULT x the action coef in both phases (see
    trainer.Trainer.run_update).
    """
    frac = min(1.0, steps / C.ENTROPY_DECAY_STEPS)
    return (C.DURATION_ENTROPY_COEF_START
            + (C.DURATION_ENTROPY_COEF_END - C.DURATION_ENTROPY_COEF_START) * frac)
