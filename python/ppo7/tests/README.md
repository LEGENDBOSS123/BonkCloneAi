# ppo7 fast checks

Both run in **seconds, on CPU, with no physics and no multiprocessing**. Run
them before starting any multi-hour pipeline phase.

```bash
cd python && PYTHONPATH=. .venv/bin/python -m ppo7.tests.test_targets_equivalence
cd python && PYTHONPATH=. .venv/bin/python -m ppo7.tests.test_trainer_smoke
```

**`test_targets_equivalence.py`** — pins every pure helper (`targets.*`,
`mirror.*`, `figar.HoldTracker`) against a reference implementation pasted
inline. These are the parts most likely to be *silently* wrong — an off-by-one
across an episode boundary, a target built from the wrong seat — and they cannot
be caught by watching a training curve. If you change a target, change the
reference too, deliberately.

**`test_trainer_smoke.py`** — drives the real `Trainer` through a stub collector
(random observations, scripted episode ends), so the whole
ingest → infer → update path executes in-process. Covers all four
configurations the pipeline actually runs:

| case | what it exercises |
|---|---|
| recurrent / phase 1 | scalar critic, dense reward, all-idle opponents, no league |
| recurrent / phase 2 | categorical critic, frozen actor |
| recurrent / phase 3 | full league — forces a snapshot **and** a complete exploiter phase |
| feedforward / TCN   | legacy conv path: both seats trained, mirror-match rows, aux heads |

Each case asserts finite losses, well-formed actions, and a checkpoint
serialize → load round-trip. `set_layout()` recomputes `STATE_DIM` and the
mirror tables, because `config.py` derives them from `RECURRENT` at import time
and the test flips it at runtime.
