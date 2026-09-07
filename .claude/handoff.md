# Handoff — ppo7 minGRU rework

## Current progress
- **minGRU recurrent encoder fully wired into ppo7 and validated** (all 5 stages
  tested). Replaces the earlier TCN window; single-frame obs + hidden state carried
  across decisions; truncated-BPTT update with `BURN_IN=4`.
- Also in this session: **FiGAR hold bonus** (`HOLD_BONUS=0.01`, learns *when* to
  hold), **adaptive entropy controller** (targets H=2.5, league-phase-gated),
  **MPS** is now the fast device, and a **league-freeze crash fix**.
- **A from-scratch gg2 pipeline is RUNNING** on MPS — phase 1, **~49–51k steps/s**
  (≈3× the TCN), winrate climbing toward the 93% gate, no errors.
- Baseline (TCN + hold-bonus, ELO 1303) saved at
  `models/gg2-baseline-elo1303-holdbonus0.json`.
- Full spec: `python/ppo7/ARCHITECTURE.md`.

## Files touched
- `python/ppo7/mingru.py` — new: `MinGRUNet` (step / forward_seq / records).
- `python/ppo7/ppo.py` — recurrent nets; `act_step`, `act_actions_step`,
  `update_recurrent`; hold-bonus term + `mean_hold`; `adapt_entropy`.
- `python/ppo7/env.py` — single-frame `decision_state`; `_raw_frame` /
  `_fourier_expand` split.
- `python/ppo7/train.py` — hidden carry + reset; `_infer_recurrent`; recurrent
  `run_update` branch (learner seat only, per-env mirror); `hold=` log; adaptive-
  entropy league gating.
- `python/ppo7/league.py` — `_make_net` / `_net_from_records` (freeze correct class;
  fixed the phase-3 snapshot crash).
- `python/ppo7/policy.py`, `python/ppo7/play.py` — recurrent eval/play policy.
- `python/ppo7/config.py` — `RECURRENT`, `MINGRU_HIDDEN=64`, `BURN_IN=4`,
  conditional `STATE_DIM`, conditional mirror, `HOLD_BONUS=0.01`, adaptive-entropy
  constants.
- `python/run_gg2_pipeline.sh` — recurrent config asserts.
- Docs: `python/ppo7/ARCHITECTURE.md` (new), this file.

## Tests / blockers
- **No failing tests.** Validated in isolation: minGRU `step`==`forward_seq` (~1e-8),
  records roundtrip exact, BPTT grad reaches GRU, mirror equivalence exact, recurrent
  `update` finite + grad flow, eval/play hidden carry + reset. Validated end-to-end:
  full recurrent pipeline runs, updates firing, no errors, ~49–51k steps/s MPS.
- **No blockers.** Open items (not blocking):
  1. `hold=` climbs high even at `HOLD_BONUS=0.01` (recurrent amplifies it) — watch;
     drop to 0.005/off if pathological in phase 3.
  2. Adaptive entropy targets *combined* H → mild coupling with the hold bonus
     (clean fix later: target action entropy alone).
  3. `play_map.mjs` deploy is unwired for recurrent (minGRU cell + hidden carry +
     `type:"mingru"` reader + new parity fixture) — only needed to ship a model.
  4. Multi-day from-scratch run in progress (arch change → TCN baseline can't load).

## Immediate next 3 actions
1. Let phase 1 hit its gate (or `kill -INT` the main to force-advance), then watch
   `hold=` through p2 → p3. If `hold=` stays pathological in p3, lower `HOLD_BONUS`
   to 0.005.
2. At the first phase-3 checkpoint, run
   `python -m ppo7.eval_h2h --a <p3 ckpt> --b models/gg2-baseline-elo1303-holdbonus0.json`
   to confirm the minGRU net is competitive (ELO is unreliable; h2h is the arbiter).
   Note: both sides must share the encoder mode — compare minGRU-vs-minGRU once a
   second checkpoint exists; the TCN baseline is a different-shape reference only.
3. If/when deploying: wire `play_map.mjs` for recurrent (minGRU cell, per-env hidden
   carry + episode reset, `type:"mingru"` records) and regenerate the parity fixture.
