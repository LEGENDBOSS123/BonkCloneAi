"""Phase presets and dotted-path overrides — pure `config -> config` transforms.

The from-scratch curriculum runs in three phases, and ppo7 expressed the
differences two ways, both bad:

* `train.apply_phase_flags` MUTATED the config module at runtime, so a
  spawned worker could in principle read different values than the parent; and
* the pipeline shell script ran
  ``sed -i '' 's/^ENTROPY_COEF_START = 0.1$/...0.01/' ppo7/config.py``
  between phases — editing source mid-run.

Here a phase is a function. Presets compose, are unit-testable
(``validate(phase1(default()))``), and can be diffed against the default. The
pipeline passes ``--set entropy.coef_start=0.01`` instead of editing a file.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable

from .config import Ppo9Config, validate

Transform = Callable[[Ppo9Config], Ppo9Config]


def default() -> Ppo9Config:
    """The phase-3 (full league) configuration — the steady state."""
    return Ppo9Config().resolve()


def phase1(cfg: Ppo9Config) -> Ppo9Config:
    """Learn to move and to kill something that cannot fight back.

    Scalar critic (the outcome classifier has nothing to classify yet), dense
    approach reward, every opponent idle, no league. Gate: winrate vs idle.
    """
    return replace(
        cfg,
        critic=replace(cfg.critic, categorical=False, atoms=None,
                       aux_gamma_coef=0.0),
        shaping=replace(cfg.shaping, dense_reward=True),
        league=replace(cfg.league, enabled=False, opp_idle_prob=1.0,
                       opp_current_prob=0.0, opp_pfsp_prob=0.0),
        ppo=replace(cfg.ppo, freeze_actor=False),
    ).resolve()


def phase2(cfg: Ppo9Config) -> Ppo9Config:
    """Freeze the phase-1 actor and fit a fresh CATEGORICAL critic to it.

    Training the classifier against a moving policy from scratch wastes the
    warm start; freezing the actor makes the value target stationary. Gate:
    critic loss converged.
    """
    return replace(
        cfg,
        critic=replace(cfg.critic, categorical=True, atoms=None),
        shaping=replace(cfg.shaping, dense_reward=False),
        league=replace(cfg.league, enabled=False),
        ppo=replace(cfg.ppo, freeze_actor=True),
    ).resolve()


def phase3(cfg: Ppo9Config) -> Ppo9Config:
    """Unfreeze everything and run the full league. Open-ended."""
    return replace(
        cfg,
        critic=replace(cfg.critic, categorical=True, atoms=None),
        shaping=replace(cfg.shaping, dense_reward=False),
        league=replace(cfg.league, enabled=True),
        ppo=replace(cfg.ppo, freeze_actor=False),
    ).resolve()


_PRESETS: dict[int, Transform] = {1: phase1, 2: phase2, 3: phase3}


def preset(phase: int) -> Transform:
    """The transform for a phase number."""
    if phase not in _PRESETS:
        raise ValueError(f"no such phase {phase}; expected one of {sorted(_PRESETS)}")
    return _PRESETS[phase]


def _coerce(current: Any, raw: str) -> Any:
    """Parse `raw` to whatever type `current` already is."""
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"cannot read {raw!r} as a bool")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))          # accept 2e6 for an int field
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        parts = [p for p in raw.strip("()[] ").split(",") if p.strip()]
        elem = current[0] if current else 0.0
        return tuple(_coerce(elem, p) for p in parts)
    if current is None or isinstance(current, str):
        return raw
    raise ValueError(f"unsupported override target of type {type(current).__name__}")


def set_path(cfg: Ppo9Config, path: str, raw: str) -> Ppo9Config:
    """Return a copy of `cfg` with one dotted field replaced.

    ``set_path(cfg, "entropy.coef_start", "0.01")``. The value is coerced to
    the field's existing type, so a typo'd path or an unparseable value fails
    loudly here rather than silently doing nothing.
    """
    parts = path.split(".")
    node: Any = cfg
    for i, key in enumerate(parts[:-1]):
        if not is_dataclass(node) or key not in {f.name for f in fields(node)}:
            raise ValueError(f"no config field at {'.'.join(parts[:i + 1])!r}")
        node = getattr(node, key)
    leaf = parts[-1]
    if not is_dataclass(node) or leaf not in {f.name for f in fields(node)}:
        raise ValueError(f"no config field at {path!r}")
    value = _coerce(getattr(node, leaf), raw)

    # Rebuild the chain bottom-up: frozen dataclasses replace, never mutate.
    def rebuild(obj: Any, keys: list[str]) -> Any:
        if not keys:
            return value
        head, rest = keys[0], keys[1:]
        return replace(obj, **{head: rebuild(getattr(obj, head), rest)})

    return rebuild(cfg, parts)


def apply_overrides(cfg: Ppo9Config, overrides: list[str]) -> Ppo9Config:
    """Apply ``["a.b=1", "c.d=x"]`` in order."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form path=value")
        path, raw = item.split("=", 1)
        cfg = set_path(cfg, path.strip(), raw)
    return cfg


def resolve(phase: int | None = None, overrides: list[str] | None = None,
            base: Ppo9Config | None = None) -> Ppo9Config:
    """`default -> preset -> overrides -> resolve -> validate`, in that order."""
    cfg = base if base is not None else default()
    if phase is not None:
        cfg = preset(phase)(cfg)
    cfg = apply_overrides(cfg, overrides or [])
    cfg = cfg.resolve()
    validate(cfg)
    return cfg
