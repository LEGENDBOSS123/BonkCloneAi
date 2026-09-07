# ppo7 — implementation spec (recurrent minGRU training stack)

Orientation for the ppo7 stack as it stands after the minGRU rework. Complements
`../../CLAUDE.md` (which still describes the pre-rework architecture). The two hard
constraints from CLAUDE.md still hold: **no state-dependent rule rewards**, and a
**minimal map-agnostic observation**.

## 1. Overview
ppo7 trains a Bonk.io-clone 1v1 brawler with PPO + an AlphaStar-lite league, on the
`gang-grounds-2-0` (gg2) map, on **MPS** (Apple GPU — see benchmarks). The temporal
encoder has two modes behind `config.RECURRENT`:
- **minGRU** (`RECURRENT=True`, default) — recurrent hidden state carried across
  decisions; single-frame observation.
- **TCN window** (`RECURRENT=False`, legacy) — dilated causal conv over a 16-frame
  raw window; retained in the code but unused.

## 2. Observation
- Raw frame = **33 dims** (`RAW_FRAME_DIM`): self 12 / opp 12 / rel 4 / pending 5.
  Fourier block = **96** (`FOURIER_BLOCK` = 12 base dims × 8, L=4 octaves k4–7).
- **Recurrent**: `STATE_DIM = 129` (single frame = raw33 + fourier96). Temporal
  memory lives in the minGRU hidden state, *not* the observation.
- **TCN**: `STATE_DIM = 1152` (16 raw frames + one current fourier block).
- `AGENT_STATE_DIM = STATE_DIM + 1` — the trainer appends the FiGAR remaining-hold
  feature (normalised) so a mid-hold state is Markov.
- Mirror augmentation is **layout-conditional** in `config.py`: recurrent = 30
  negate / 3 swap indices; windowed = the same map tiled across frames. Verified
  `mirror(expand(x)) == expand(mirror(x))` (exact) for both.

## 3. minGRU encoder (`mingru.py`)
Chosen over GRU/LSTM (sequential BPTT, saturating gates → unstable), LRU
(complex-valued, more bug surface), and transformer (quadratic, heavy deploy).

- **Linear recurrence** `h_t = (1 − z_t)·h_{t−1} + z_t·h̃_t` with **input-only**
  gates `z, h̃ = split(Linear(x))`. Linear ⇒ stable gradients; gates for a whole
  sequence compute in one batched matmul (only the elementwise scan is sequential).
- Net = minGRU (`MINGRU_HIDDEN = 64`) → MLP `[256, 256]` (tanh) → head, with a
  **skip connection**: the MLP sees `concat([x, h])`, i.e. the current frame *and*
  the recurrent memory. ~97k params.
- Interfaces:
  - `step(x[B,in], h[B,H]) → (out, h2)` — one decision (collection / eval).
  - `forward_seq(x[B,L,in], h0, reset) → (out[B,L,·], hs)` — BPTT update; `reset[:,t]`
    zeros the incoming hidden before step t (new-episode boundary).
- Records: tfjs-style dense blocks under `type:"mingru"` (`gru` + `mlp`); rebuilt via
  `MinGRUNet.from_records`. Validated: `step`==`forward_seq` (~1e-8), records roundtrip
  exact, BPTT gradient reaches the GRU weights.

## 4. Recurrent PPO training (`ppo.py`, `train.py`)
**Hidden-state carry** (trainer, per env, `float32 [E, H]`): `h_a`/`h_c` (learner
actor/critic) and `h_opp` (seat-1 opponent). Stepped **every decision-cycle**; reset
to 0 at **episode ends** and on **phase switches** (`_reset_collection`). The pre-step
hidden is committed per row as `b_ha`/`b_hc` (only index 0 is used by the update).

**Collection** — `Trainer._infer_recurrent()`:
- Learner acts via `PPOAgent.act_step` (carries `h_a`, `h_c`).
- Opponents (current self / pool / frozen-main) act via `PPOAgent.act_actions_step`,
  which runs on the **net's own device** (pool nets are frozen on CPU).
- The minGRU steps every cycle so its hidden matches the per-cycle buffer; FiGAR
  holds only gate *which sampled action is applied*, not whether the GRU advances.

**Update** — `PPOAgent.update_recurrent()` (dispatched from `run_update` when
`self.recurrent`):
- **Truncated BPTT** over each env's whole (short) rollout, re-forwarded from the
  stored start hidden `b_ha[0]`/`b_hc[0]`. Minibatches are over **environments**.
- `BURN_IN = 4` skips loss on the leading rows (lets the recomputed hidden re-settle
  after params drift across epochs).
- Loss = clipped PPO surrogate (free rows only) + categorical-critic cross-entropy +
  hold bonus + entropy, all masked. GAE is reused unchanged (`_gae_columns`, γ=1,
  λ=0.95, `[T,E]`-shaped).
- **Mirror in the update** is per-**env** (whole trajectory), never per-row, so each
  GRU sequence stays internally consistent.

**Simplification (intentional):** recurrent mode trains the **learner seat only**.
Seat 1 still acts as a full opponent (with its own carried hidden), but its rows are
not replayed — this drops the self-play *mirror-both-train* to keep the recurrence
tractable and correct.

**Burn-in note:** the rollout is wide + short (T ≈ `ROLLOUT_STEPS/E` ≈ 22
decisions/env), so R2D2-style mid-episode burn-in *slicing* doesn't fit — a 32-step
window won't even fit in 22 steps. The correct on-policy form here (and what's
implemented) is truncated BPTT from the **stored rollout-start hidden** (hidden
carried across rollouts, reset only at episode ends); that stored hidden *is* the
bootstrap. `BURN_IN` is only the small post-epoch-staleness skip.

## 5. Hold bonus — FiGAR temporal abstraction (`config.HOLD_BONUS`, `ppo.py`)
- `HOLD_BONUS = 0.01`: a **state-independent** `+β·E[log2 hold]` term added to the
  **actor loss** — *not* the reward, so the categorical critic stays an exact 3-atom
  classifier. Per-duration weight `w_d = log2(hold) = duration index`.
- Goal: the agent learns *when* to hold keys (coherent temporally-extended
  exploration + fewer decisions/game). The **return gradient supplies the "when"**
  (it pushes holds short wherever re-deciding matters); β only wins where holding is
  free. Constraint-safe (gamma/time-cost class, state-independent).
- Logged as `hold=` (mean sampled hold in cycles, free rows) in the train line.
- **Tuning:** `eval_h2h` flat + `hold=` up = "free" holds. The recurrent net
  **amplifies** this: `0.03` drove `hold≈43` from scratch; `0.01` is the chosen
  start and still climbs (~20 in early phase 1) — **watch it**, drop to 0.005/off if
  it stays pathological in phase 3.

## 6. Adaptive entropy controller (`PPOAgent.adapt_entropy`, `train.py`)
- SAC-style automatic temperature: holds the **combined** (action+duration) entropy
  H at `ENTROPY_TARGET_H = 2.5` by nudging `ent_coef` (× / ÷ `ENTROPY_ADAPT_RATE =
  1.02`, clamped `[ENTROPY_COEF_MIN 0.001, ENTROPY_COEF_MAX 0.20]`, init
  `ENTROPY_COEF_INIT 0.015`). Duration coef `dec = 2·ec` **always**.
- **Gated to the league phase** (`C.LEAGUE_ENABLED`): phases 1/2 keep the plain
  scheduled coef (`entropy_coef_at`) so their gates aren't held back.
- **Known coupling (document, not a bug):** it targets *combined* H, so the hold
  bonus (which peaks the duration head) mildly self-dampens through it. Clean future
  fix = target the *action* entropy alone.

## 7. Config reference (`config.py`)
| constant | value | meaning |
|---|---|---|
| `RECURRENT` | `True` | minGRU mode (else legacy TCN window) |
| `MINGRU_HIDDEN` | `64` | minGRU hidden width |
| `BURN_IN` | `4` | rows skipped from loss to warm the recomputed hidden |
| `STATE_DIM` | `129` (rec) / `1152` (TCN) | env obs width (conditional) |
| `AGENT_STATE_DIM` | `STATE_DIM+1` | + FiGAR remaining-hold feature |
| `HOLD_BONUS` | `0.01` | β on `E[log2 hold]` in the actor loss |
| `ENTROPY_ADAPTIVE` | `True` | SAC-style entropy controller (league phase) |
| `ENTROPY_TARGET_H` | `2.5` | target combined entropy (nats) |
| `ENTROPY_ADAPT_RATE` | `1.02` | per-update ec multiplier |
| `ENTROPY_COEF_INIT/MIN/MAX` | `0.015 / 0.001 / 0.20` | controller start + clamps |
| `ENTROPY_COEF_START` | `0.1` | phase-1 scheduled coef; pipeline seds → 0.01 pre-phase-3 |
| `TCN_WINDOW/DILATIONS/CHANNELS/EMBED` | `16 / [1,2,4,8] / 16 / 64` | legacy TCN (unused when `RECURRENT`) |

## 8. Module map
The stack is split so each file owns one concern and can be read (or tested) on
its own. Arrows are "imports".

**Core training loop**
- **`train.py`** — entry point: CLI flags, phase-flag echo (the pipeline greps
  it), checkpoint writing/pruning, the run loop, and the progress line
  (`watch.py` parses it — keep `^step N` and ` wr NN%` stable).
- **`trainer.py`** — `Trainer`. One cycle = **ingest** (fold rewards, commit the
  row, league bookkeeping) → **infer** (`_infer` feedforward / `_infer_recurrent`
  minGRU) → **update** (`_update_feedforward` / `_update_recurrent`).
- **`rollout.py`** — `RolloutBuffer`: the `[T, E]` streams plus the in-flight
  row, with the two-phase `add_reward` → `commit` protocol a decision-level env
  forces (the reward for row *t* only exists at cycle *t+1*).
- **`figar.py`** — `HoldTracker`: one seat's "which action is latched, for how
  much longer". Three call shapes (whole seat / mirror-match subset / per-net
  group), all through `free_mask` + `latch` + `advance`.
- **`targets.py`** — pure array→array learning targets: `gae_columns`,
  `outcome_targets`, `time_targets`, `aux_targets`, `approach_shaping`. No
  trainer state, no torch — unit-testable against a reference in milliseconds.
- **`schedules.py`** — the open-loop entropy schedules (the closed-loop
  controller is `PPOAgent.adapt_entropy`).

**Agent + nets**
- **`ppo.py`** — `PPOAgent`: acting (`act_batch` / `act_step` /
  `act_actions_step`), both update paths, the hold bonus, the adaptive-entropy
  controller, serialization.
- **`mingru.py`** — `MinGRUNet` (`step` / `forward_seq` / records). Default.
- **`networks.py`** — `TemporalNet` (legacy TCN) and `MLP`.

**Environment**
- **`env.py`** — `LagEnv`: netcode/lag simulation, the raw frame, Fourier
  expansion, `decision_state` in either layout. Re-exports the mirror helpers.
- **`mirror.py`** — the horizontal-mirror transform for obs and actions (the
  index tables themselves are layout-dependent and live in `config.py`).
- **`collect.py`** — `VecCollector`: shared-memory multiprocess workers; the
  main process never sees a tick, only decision blocks.
- **`config.py`** — every constant, including the layout-conditional
  `STATE_DIM` / mirror tables.

**Population**
- **`league.py`** — snapshots, exploiters, PFSP sampling, the exploiter phase
  machine, and `_make_net` / `_net_from_records` (frozen pool members must match
  the live actor's class or the state_dict keys mismatch).

**Off the training path**
- **`policy.py`** — `ActorPolicy` (alias `MLPPolicy`): rebuilds a frozen actor
  from records, enforces FiGAR holds and carries recurrent memory, so eval and
  play see exactly what the trainer saw. `resolve_agent` picks main/snap/exp.
- **`eval_h2h.py`** — head-to-head between two checkpoints. **The progress
  metric**; ELO is a treadmill.
- **`play.py`** — pygame playground. **`watch.py`** — log → progress file.
- **`gen_spawns.py`** — pre-probes the survivable-spawn pool for a map.
- **`movement.py` / `pretrain_move.py` / `actor.py` / `hca.py`** — earlier
  experiments (movement-skill transplant, hindsight credit assignment). Not
  imported by the training path.

## 9. Benchmarks (measured this session)
- Learner update (90k rows, 4 epochs): TCN **6.5s CPU / 7.6s MPS**; minGRU
  **4.5s CPU / 2.4s MPS**. Inference/cycle: minGRU **~12 ms** vs TCN ~50 ms.
- End-to-end: TCN ch16 ~15–17k steps/s (MPS); **minGRU ~49–51k steps/s (MPS)**,
  phase 1. minGRU's classic-GRU BPTT slowness does not appear (parallel gates).
- **MPS beats CPU** for both nets at this size (the pre-rework "train on CPU" rule
  inverted once the net + batches grew). Microbench overstates the MPS inference win
  vs the real loop (per-cycle CPU→GPU transfer).

## 10. Run / eval / resume
- **Pipeline:** `./run_gg2_pipeline.sh` — phase 1 (kill-idle, dense, scalar critic)
  → phase 2 (freeze actor, warm categorical critic) → phase 3 (full league +
  adaptive entropy). MPS, auto-advances on gates (p1 wr≥93%, p2 critic converged,
  floor 200M steps). **Force-advance** a phase = `kill -INT` the phase's main
  process (it saves a `-final` checkpoint the pipeline picks up).
- **Eval (the real progress metric — not ELO):**
  `python -m ppo7.eval_h2h --a <new>.json --b <old>.json --games 2000`.
- **Baseline** TCN + hold-bonus model kept at
  `../../models/gg2-baseline-elo1303-holdbonus0.json` (incompatible with minGRU by
  shape; comparison via eval only, both must share the encoder mode).
- **Resume:** `--load <ckpt>` (same encoder mode). Hidden state is not serialized —
  it re-warms from zero on resume.
- **Deploy (`../../play_map.mjs`) is UNWIRED for recurrent** — needs a minGRU cell +
  per-env hidden carry + episode reset + a `type:"mingru"` records reader, then a
  regenerated parity fixture. Only needed to ship a trained model.
