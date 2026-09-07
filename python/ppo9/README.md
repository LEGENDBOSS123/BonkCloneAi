# ppo9

A clean rebuild of the ppo7 training stack, implementing the keep/drop verdicts
in `../ppo7/PPO_FEATURES.md`. The environment layer lives in `../bonkenv/`,
shared and trainer-agnostic.

**No new capabilities.** ppo9 is ppo7 minus six rejected features, minus the
dead conv-encoder path, restructured onto injected dataclass configs.

## Quick start

```bash
cd python && source .venv/bin/activate

python -m ppo9.config --dump                 # the effective config, as JSON
python -m ppo9.tests.test_ppo9_smoke         # full wiring check, ~4 s

./run_gg9_pipeline.sh                        # from-scratch 3-phase run
python -m ppo9.eval_h2h --a <new>.json --b <old>.json --games 2000
python -m ppo9.play --checkpoint <ckpt>.json # you vs it
python -m ppo9.watch ../runs/gg9-p3/train.log --out progress.txt
```

## Layout

| file | owns |
|---|---|
| `config.py` | frozen nested dataclasses, `resolve()`, `validate()` |
| `presets.py` | phase transforms and `a.b=value` overrides |
| `mingru.py` / `nets.py` | the recurrent encoder and its records format |
| `figar.py` | hold tracking (dynamic action repetition) |
| `schedules.py` | open-loop entropy schedules |
| `rollout.py` | the dense `[T,E]` buffer and its two-phase commit |
| `targets.py` | pure array→array learning targets |
| `losses.py` | pure tensor→tensor loss terms |
| `agent.py` | `PPOAgent`: acting, the BPTT update, persistence |
| `league.py` | snapshots, PFSP, the exploiter phase machine |
| `trainer.py` | one collection cycle, and the update around it |
| `train.py` | CLI, checkpoints, the progress line |
| `policy.py` / `eval_h2h.py` / `play.py` / `watch.py` | off the training path |

## What changed versus ppo7

**Removed:** RUDDER · HCA · the hold bonus · the adaptive entropy controller ·
the position/time auxiliary heads · tree search · the TCN/feedforward path ·
param-noise exploiter exploration · the derived duration-entropy coefficient.

**Three correctness fixes** the rebuild made possible:

1. **One `Outcome` ordering.** `ppo7/env.py` used `WIN, LOSS, DRAW = 0, 1, 2`
   while its rollout, targets and trainer all used win/draw/loss. It survived
   only because the env's value was discarded at the worker boundary. One
   `IntEnum` now crosses that boundary.
2. **Reward values and critic atoms are the same table, per matchup.** ppo7
   kept them apart and they disagreed: an exploiter's critic ran on a 2.0 win
   scale while the env paid it 1.0, so a *certain win* produced an advantage of
   `1.0 - 2.0 = -1.0`. `validate()` now rejects any config where they differ.
3. **`b_ha` is one row, not `T`.** Only the rollout-start actor hidden was ever
   read.

**Structural:** no module-level config (so two layouts can train side by side in
one process — see `tests/test_ppo9_smoke.py`), no runtime config mutation, and
no `sed`-editing of source between pipeline phases.

## Reading the progress line

```
step 46,341,120 (ep 167,894) [MAIN|snap 0 exp 0] | steps/s 38615 | ELO 1000.0 |
wr 72% | topWR 61% (snap18) expTop 56% lose 22 gap 112k | updates 452 |
a=-0.2124 c=0.0405 H=2.918 (a=1.25/d=1.66) ec=0.099 kl=0.0226 clip=3%/1%
hold=33.4 decided=34% epT=189 g2=0.0412 | perf/cycle ...
```

| field | meaning |
|---|---|
| `wr` | the MAIN's rolling score over its last 100 episodes (draws count half) |
| `H (a=/d=)` | combined entropy, split into the action and duration heads |
| `kl` | approx KL per update, **over free rows only** |
| `clip=X%/Y%` | X = free rows whose ratio left the trust region (the usual "clipfrac"); Y = rows where the clip actually zeroed the gradient. Y ≤ X always; the two converging means the clip is genuinely restraining updates, X high with Y low means the policy is moving a lot but mostly unimpeded |
| `klstop=N%` | only when `ppo.kl_stop > 0`: share of minibatches whose actor step was skipped for exceeding it |
| `hold` / `decided` | mean sampled hold in cycles, and the share of rows that were real decisions. A low `decided%` with a rising `hold` is the passive-drift signature |
| `epT` | mean decisions per episode |
| `g2` | multi-gamma auxiliary loss, when those heads are on |

Every per-row statistic is masked to FREE decision rows. That is not cosmetic:
on a held row the stored log-prob belongs to an action that was sampled and
then **discarded** in favour of the latched one, so any ratio computed there is
meaningless. An unmasked `kl` on a rollout at `decided=3%` reads ~4.7 — a
fabricated trust-region violation — against a true value of ~0.02.

## Tests

All CPU, all seconds, no physics unless noted.

| file | covers |
|---|---|
| `tests/test_nets.py` | `step` ≡ `forward_seq`, records round-trip, BPTT gradient, config pickling through `spawn` |
| `tests/test_targets.py` | every target vs an inline reference **and** vs ppo7 |
| `tests/test_update.py` | the backward pass: categorical, scalar+clipped, frozen actor, aux gamma heads |
| `tests/test_ppo9_smoke.py` | all three phases end to end through a stub collector, incl. a full exploiter phase |

`../bonkenv/tests/test_ppo7_parity.py` proves the env port is bit-exact against
ppo7. Delete it, and the ppo7 cross-checks in `test_targets.py`, when ppo7 is
retired.
