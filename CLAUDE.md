# CLAUDE.md — orientation for AI assistants working in this repo

This is a self-play RL project that trains an agent to play a Bonk.io-clone 1v1
brawler. Read this before making changes; the constraints below are load-bearing
design decisions, not accidents.

## The two hard constraints (do not violate without being asked)
1. **Sparse terminal rewards only.** `+1`/`−1`/`−0.5` at round end, nothing else
   (`config.py`: `WIN_REWARD`/`LOSS_REWARD`/`DRAW_REWARD`). No reward shaping,
   no intrinsic bonuses, no potential-based tricks — the whole experiment is
   "can sparse rewards + minimal input beat humans."
2. **Minimal, map-agnostic observation.** 30 floats, no hand-coded features
   specific to a map. See the obs layout in `env2.py` (self 10 / opp 10 / rel 4 /
   pending 5 / draw-clock 1), all normalized.

## Where things are
- `python/bonk/` is the live project. The JS in `src/` is the original game +
  earlier in-browser TF.js training (`src/rl`, `src/rl2`), now superseded.
- `sim.py` — Box2D (`Box2D==2.3.10`, needs **Python 3.12**) port of the JS
  physics, verified to match the JS sim to ~1e-5 m. If you touch it, re-verify
  with `verify_replay.py`; behavioral parity is the reason the whole pipeline works.
- `env2.py` — `LagEnv`: decision-level env with **input-lag simulation**
  (a decision at tick `t` applies at `t+lag`; `lag` randomized per episode). This
  is the sim-to-real bridge; don't remove it.
- `ppo.py` — `PPOAgent`: PPO over 18 joint actions (categorical), `[256,256]` MLP
  with LayerNorm. Single source of truth for `HIDDEN`, entropy schedule, LRs.
- `train_ppo_mp.py` — **the trainer people actually run.** Multiprocess collection
  + AlphaStar-lite league (main phase / exploiter phase, snapshot pool, exploiter
  pool, PFSP opponent sampling). Start here to understand training dynamics.
- `train_ppo.py` — single-process version; also exports shared constants
  (`GAMMA`, `SNAPSHOT_*`, `compute_gae`) that the MP trainer imports.
- `mp_collect.py` — shared-memory vectorized collector (workers own envs).
- `networks.py` — torch MLP with `to_records()`/`from_records()` in a TF.js-
  compatible layout (kernels transposed, LayerNorm eps 1e-3). `from_records`
  **infers architecture from weight shapes**, so deploy/eval work at any hidden size.
- `eval_h2h.py` — head-to-head between two checkpoints. **Use this, not ELO, to
  judge progress** (ELO is measured vs a rolling buffer of recent selves and
  plateaus regardless of real improvement).
- `export_model.py` — strips a 100+ MB checkpoint to a ~1 MB actor-only JSON.
- `play.py` — pygame playground (human vs agent). `../play2.mjs` — deploy to the
  real game; rebuilds the net in TF.js from the actor records.

## Running things (from `python/`, venv active)
```bash
python -m bonk.train_ppo_mp --workers 8            # train (CPU; MPS is ~100x slower here)
python -m bonk.train_ppo_mp --load runs/ppo-mp/<ckpt>.json   # resume
python -m bonk.eval_h2h --a <new>.json --b <old>.json --games 2000
python -m bonk.export_model runs/ppo-mp/<ckpt>.json --out ../models/bonk-ppo-latest.json
python -m bonk.play --load runs/ppo-mp/<ckpt>.json --net actor   # pygame, human vs AI
```
Notes: train on **CPU** (tiny nets → per-decision dispatch dominates on MPS/GPU).
Pipe through `python -u` when capturing logs. Checkpoints are JSON in `runs/`.

## League training model (the core algorithm)
- **Main phase**: the saved agent trains against a uniform-ish mix of {current
  self, past snapshot, exploiter}, where snapshots *and* exploiters are sampled by
  **PFSP** — weighted toward the ones the main currently *loses* to (`_snap_weight`,
  `_exp_weight`, tracked via each opponent's `recent` deque).
- **Exploiter phase**: every `EXPLOITER_INTERVAL` main episodes, main training
  freezes and a fresh exploiter best-responds to the frozen main to find a hole;
  it graduates on a winrate gate (+ sharpen tail) or a timeout, then joins the pool.
- Key tunables live at the top of `train_ppo_mp.py` (`EXPLOITER_*`) and in
  `ppo.py`/`train_ppo.py` (`HIDDEN`, entropy schedule, `GAMMA`, `SNAPSHOT_*`).

## Known open problems (as of writing)
- The agent **cycles** (rock-paper-scissors) rather than monotonically improving;
  `eval_h2h` of current-vs-1M-ago is the diagnostic. PFSP over snapshots was added
  to counter this.
- **Juke vulnerability**: it rushes and falls off the edge when the opponent dodges.
  Leading hypotheses: (a) a strategic-basin problem the league must pressure it out
  of, (b) a perception limit — a single frame can't show motion/intent under input
  lag, which a **frame-stack** (last 2–3 obs) would fix (but changes `STATE_DIM` →
  retrain + update `play2.mjs`).
- PPO explores at the action level, which is weak for the *strategic* shifts (rush →
  spacing) this game needs.

## Gotchas
- Changing `STATE_DIM` / obs layout means retraining from scratch and updating the
  obs construction in `play2.mjs` and `play.py` to match.
- `MAX_EPISODE_STEPS` and `ACTION_REPEAT` must match between `config.py` and
  `play2.mjs` or the draw-clock feature and decision cadence desync at deploy.
- Old checkpoints won't load into an agent with a different `HIDDEN` size (shape
  mismatch), but deploy/eval infer size from records and are fine.
