"""Entry point for resuming a run with sparse search-augmented collection (the
"m0" technique: `LookaheadPolicy.search()` with no continuation, `m_rollout_
cycles=0` -- K candidates, apply, score directly) folded into a small percentage
of the learner's free decisions.

    python -m ppo7.train_search_augmented --load <ckpt.json> --out <dir>

Mirrors `train.py`'s CLI/checkpoint/logging conventions (reuses its
`CheckpointWriter` and `progress_line` directly) so this run is inspectable
the same way the rest of the pipeline is -- the only substantive difference is
the collector (`SingleProcessCollector`, not multiprocess `VecCollector`) and
the trainer (`SearchAugmentedTrainer`, not plain `Trainer`).

WHY SINGLE-PROCESS. `SearchAugmentedTrainer` needs direct access to the live
`LagEnv` objects to fork/restore around search candidates
(`LagEnv.snapshot_state`/`restore_state`), which the real multiprocess
`VecCollector` never exposes to the trainer process by design (workers own
their envs; the main process only ever sees shared obs/act arrays). Wiring
search into that architecture would need shipping weights to workers
(currently never done) and new worker-side search code -- a substantially
bigger, riskier build than this file. Single-process trades throughput
(measured ~16-17k steps/s at E=128 here, vs. ~38k steps/s for the real
8-worker/2560-env pipeline -- about 2.2x slower, not the order-of-magnitude
worse a naive single-process port would be, since a tiny net's per-cycle cost
is dominated by physics ticking, which plateaus around E=128 on this machine)
for running TODAY on code that's already been correctness-tested this
session (bit-exact snapshot/restore, bit-exact collector protocol, a real
Trainer running against it, and this exact League+SearchAugmentedTrainer
combination smoke-tested against a real large checkpoint before launch).
"""

import argparse
import json
import signal
import time
from pathlib import Path

import torch

from . import config as C
from .experiment_search_finetune import SearchAugmentedTrainer
from .ppo import PPOAgent
from .single_process_collector import SingleProcessCollector
from .train import CheckpointWriter, progress_line

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True, help="checkpoint to resume from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=128,
                    help="single-process env count; steps/s plateaus ~128 here")
    ap.add_argument("--steps", type=int, default=100_000_000_000)
    ap.add_argument("--save-every", type=int, default=30_000_000,
                    help="checkpoint interval in env steps (~30min at measured "
                    "throughput)")
    ap.add_argument("--search-frac", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth-cap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=0)
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent(C.AGENT_STATE_DIM, C.NUM_ACTIONS, device="cpu")
    coll = SingleProcessCollector(args.envs)
    trainer = SearchAugmentedTrainer(
        coll, agent, search_frac=args.search_frac, k=args.k,
        depth_cap_cycles=args.depth_cap, m_rollout_cycles=0, seed=args.seed)
    save = CheckpointWriter(trainer, out_dir).save

    with open(args.load) as f:
        trainer.load_state(json.load(f))
    print(f"bonk2/ppo search-augmented (m0): {args.envs} envs (single-process), "
         f"search_frac={args.search_frac}, k={args.k}, depth_cap={args.depth_cap}, "
         f"m_rollout=0, hidden {C.HIDDEN}")
    print(f"resumed: episode {trainer.episode_count}, env_steps {trainer.env_steps}, "
         f"ELO {trainer.league.current_rating:.1f}, "
         f"snapshots {len(trainer.league.snapshots)}, "
         f"exploiters {len(trainer.league.exploiters)}")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    t_start = time.perf_counter()
    last_log, last_steps, last_saved = t_start, trainer.env_steps, trainer.env_steps
    last_perf = dict(trainer.perf)
    last_stats = None

    try:
        while trainer.env_steps < args.steps and not stop["flag"]:
            trainer.cycle()
            if trainer.learner_steps() >= C.ROLLOUT_STEPS:
                last_stats, _ = trainer.run_update()

            now = time.perf_counter()
            if now - last_log >= 5.0:
                sps = (trainer.env_steps - last_steps) / (now - last_log)
                line = progress_line(trainer, last_stats, sps, last_perf, now, last_log)
                print(line + f" | searched {trainer.searched_rows}/"
                      f"{max(1, trainer.free_rows)} "
                      f"({trainer.searched_rows/max(1,trainer.free_rows)*100:.2f}%)")
                last_log, last_steps = now, trainer.env_steps
                last_perf = dict(trainer.perf)

            if trainer.env_steps - last_saved >= args.save_every:
                last_saved = trainer.env_steps
                save()
    except Exception:
        save("-crash")
        raise
    save("-final" if stop["flag"] else "")
    print(f"done: {trainer.episode_count} episodes in "
         f"{(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
