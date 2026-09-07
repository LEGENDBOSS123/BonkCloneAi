"""bonkenv — the Bonk.io-clone 1v1 environment, shared across trainers.

Everything about the GAME lives here: physics backend, netcode/lag simulation,
observation layout, spawn distribution, and multiprocess collection. Nothing
about LEARNING does — no network shapes, no PPO hyperparameters, no league.

Hard rule: this package imports only `numpy`, `bonk` and `bonk3`. It must never
import `torch` or any trainer package, so a collection worker never pays for a
deep-learning framework it does not use. `tests/test_no_torch.py` enforces it.
"""
from .actions import IDLE, NUM_ACTIONS, Outcome, index_to_keys, mirror_action, mirror_action_batch
from .collect import (DONE_ARCHIVE, DONE_ENDED, DONE_OUTCOME, DONE_TIMEOUT,
                      DONE_WIDTH, NO_ARCHIVE, ArchiveBroadcast, ArchiveRequest,
                      ArchiveSnapshot, VecCollector)
from .config import (CurriculumConfig, EngineConfig, EnvConfig, EpisodeConfig,
                     LagConfig, ObsConfig, OutcomeValues, SpawnConfig, validate)
from .env import LagEnv, StepResult
from .layout import ObsLayout, build_layout
from .mirror import mirror_obs, mirror_obs_batch
from .obs import BLOCK_DIM, OPP_OFF, PEND_OFF, REL_OFF, SELF_OFF, fourier_expand
from .spawns import SpawnCurriculum, SpawnPool

__all__ = [
    "IDLE", "NUM_ACTIONS", "Outcome", "index_to_keys",
    "mirror_action", "mirror_action_batch", "mirror_obs", "mirror_obs_batch",
    "EnvConfig", "EngineConfig", "ObsConfig", "LagConfig", "EpisodeConfig",
    "SpawnConfig", "CurriculumConfig", "OutcomeValues", "validate",
    "ObsLayout", "build_layout", "LagEnv", "StepResult",
    "SpawnPool", "SpawnCurriculum", "VecCollector",
    "DONE_ENDED", "DONE_TIMEOUT", "DONE_OUTCOME", "DONE_ARCHIVE", "DONE_WIDTH",
    "NO_ARCHIVE", "ArchiveRequest", "ArchiveSnapshot", "ArchiveBroadcast",
    "SELF_OFF", "OPP_OFF", "REL_OFF", "PEND_OFF", "BLOCK_DIM", "fourier_expand",
]
