"""ppo7 entry point: CLI, checkpoint I/O, the training loop, and the log line.

    python -m ppo7.train --workers 8 --envs-per-worker 320 --device mps

Workers collect (`collect.VecCollector`), the main process infers, learns and
does the bookkeeping (`trainer.Trainer`); all population logic lives in
`league.League`. This module is deliberately thin — it owns only the things a
run needs from the outside: the flags, the checkpoint files, and what gets
printed. The `run_gg2_pipeline.sh` phases drive training entirely through these
flags, and parse the phase-flags line and the step line back out of the log.

Device strategy (measured on Apple Silicon at E=2560, minGRU [256,256]): the PPO
update and whole-batch inference are both faster on MPS, so --device mps puts
the trained agents (main / exploiter / frozen_main) there; frozen POOL nets
always stay on CPU, where many small per-net batches beat GPU dispatch overhead.
Plain --device cpu remains fully supported.

Ctrl-C saves a `-final` checkpoint and shuts the workers down — that is also how
the pipeline force-advances a phase.
"""

import argparse
import json
import signal
import time
from pathlib import Path

import torch

from . import config as C
from .collect import VecCollector
from .ppo import PPOAgent
from .trainer import Trainer

REPO = Path(__file__).resolve().parents[2]


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"))
    ap.add_argument("--load")
    ap.add_argument("--warm-actor",
                    help="load ONLY the agent weights from this checkpoint "
                         "(fresh league, progress reset) -- for a map switch")
    ap.add_argument("--out", default=str(REPO / "runs/ppo2"))
    ap.add_argument("--steps", type=int, default=0,
                    help="stop after this many ENV STEPS (primary unit)")
    ap.add_argument("--episodes", type=int, default=100_000_000,
                    help="secondary cap; --steps takes precedence when set")
    ap.add_argument("--save-every", type=int, default=400_000,
                    help="checkpoint interval in ENV STEPS")
    ap.add_argument("--replay-every", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--envs-per-worker", type=int, default=320)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scalar-critic", action="store_true",
                    help="phase 1: scalar value critic instead of categorical")
    ap.add_argument("--dense-reward", action="store_true",
                    help="phase 1: dense distance-decrease reward in the return")
    ap.add_argument("--dense-coef", type=float, default=None)
    ap.add_argument("--all-idle", action="store_true",
                    help="phase 1: opponent is the stationary bot 100%% of the time")
    ap.add_argument("--freeze-actor", action="store_true",
                    help="phase 2: train only the critic, actor held fixed")
    ap.add_argument("--no-league", action="store_true",
                    help="disable snapshots + exploiters (implied by --all-idle)")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="cap torch CPU threads (0 = leave default)")
    ap.add_argument("--exploiter-trigger-wr", type=float,
                    help="override config.EXPLOITER_TRIGGER_WR")
    ap.add_argument("--exploiter-max-interval", type=int,
                    help="override config.EXPLOITER_MAX_INTERVAL")
    ap.add_argument("--exploiter-max-episodes", type=int,
                    help="override config.EXPLOITER_MAX_EPISODES")
    ap.add_argument("--force-exploiter", action="store_true",
                    help="start an exploiter phase IMMEDIATELY on launch, "
                    "bypassing the normal mastery/spacing gate -- for "
                    "deliberately testing exploiter behavior without waiting")
    ap.add_argument("--exploiter-type", choices=("random", "staller", "combat"),
                    default="random",
                    help="with --force-exploiter, pin the forced exploiter's "
                    "type instead of the normal random draw")
    return ap


def apply_phase_flags(args):
    """Mutate the config module to match the CLI, then echo the EFFECTIVE
    settings. The pipeline greps this line to verify each phase actually started
    in the configuration it asked for, so its wording is load-bearing."""
    if args.scalar_critic: C.CRITIC_CATEGORICAL = False
    if args.dense_reward:  C.DENSE_REWARD = True
    if args.dense_coef is not None: C.DENSE_COEF = args.dense_coef
    if args.all_idle:      C.OPP_IDLE_PROB = 1.0
    if args.freeze_actor:  C.FREEZE_ACTOR = True
    if args.no_league or args.all_idle: C.LEAGUE_ENABLED = False
    print(f"  phase flags: scalar_critic={not C.CRITIC_CATEGORICAL} "
          f"dense={C.DENSE_REWARD}({C.DENSE_COEF}) all_idle={C.OPP_IDLE_PROB==1.0} "
          f"freeze_actor={C.FREEZE_ACTOR} league={C.LEAGUE_ENABLED}")
    for arg, name in ((args.exploiter_trigger_wr, "EXPLOITER_TRIGGER_WR"),
                      (args.exploiter_max_interval, "EXPLOITER_MAX_INTERVAL"),
                      (args.exploiter_max_episodes, "EXPLOITER_MAX_EPISODES")):
        if arg is not None:
            setattr(C, name, arg)


class CheckpointWriter:
    """Atomic checkpoint files plus auto-pruning.

    Checkpoints embed the whole league (~780 MB each), so only the most recent
    CHECKPOINT_KEEP regular ones are kept; anything tagged (`-final`, `-crash`)
    is never pruned. Each is written to a temp file and renamed, so a kill
    mid-write cannot leave a truncated file (which broke --load twice).
    """

    def __init__(self, trainer: Trainer, out_dir: Path):
        self.trainer = trainer
        self.out_dir = out_dir

    def save(self, tag=""):
        t = self.trainer
        path = self.out_dir / (f"bonk2-ppo-step{t.env_steps}"
                               f"-ep{t.episode_count}"
                               f"-elo{round(t.league.current_rating)}{tag}.json")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(t.serialize()))
        tmp.replace(path)
        print(f"checkpoint saved: {path}")
        if not tag:
            self._prune()

    def _prune(self):
        ckpts = sorted(self.out_dir.glob("bonk2-ppo-step*.json"),
                       key=lambda q: q.stat().st_mtime)
        regular = [q for q in ckpts if "-final" not in q.name]
        for old in regular[:-C.CHECKPOINT_KEEP]:
            old.unlink(missing_ok=True)


def progress_line(trainer: Trainer, stats, sps, last_perf, now, last_log):
    """The one line per ~5 s that is the entire live view of a run. The pipeline
    parses `^step N` and ` wr NN%` out of it to decide when a phase is done, so
    keep those two shapes stable."""
    lg, p = trainer.league, trainer.perf
    dc = max(1, p["cycles"] - last_perf["cycles"])
    if stats:
        loss = (f"a={stats['actor_loss']:.4f} c={stats['critic_loss']:.4f} "
               f"H={stats['entropy']:.3f}"
               + (f" (a={stats['action_entropy']:.2f}/d={stats['duration_entropy']:.2f})"
                  if "action_entropy" in stats else "")
               + f" ec={stats['ent_coef']:.3f} hold={stats.get('mean_hold', 1.0):.1f}")
        if "decided_pct" in stats:
            # the SAME two diagnostics the original HCA main-agent test used
            # to catch a stall-collapse (config.py's HCA comment): a real drop
            # in decided% alongside a rise in mean_ep looks the same here.
            loss += f" decided={stats['decided_pct']:.0f}%"
            mep = stats.get("mean_ep_decisions", float("nan"))
            if mep == mep:   # not NaN
                loss += f" epT={mep:.0f}"
        if "hca_hdiv" in stats:
            # hdiv near 0 = h agrees with pi, HCA is contributing nothing;
            # large = pulling hard. z_acc absent until HCA_MIN_ROWS is reached.
            loss += f" hdiv={stats['hca_hdiv']:.3f}"
            if "hca_z_acc" in stats:
                loss += f" zacc={stats['hca_z_acc']*100:.0f}%"
        if "rudder_loss" in stats:
            loss += f" rud={stats['rudder_loss']:.4f}"
        if "gamma_loss" in stats:
            loss += f" g2={stats['gamma_loss']:.4f}"
    else:
        loss = "warmup"
    phase = ("MAIN" if lg.phase == "main"
             else f"EXPL {lg.phase_steps/1e6:.1f}M/{C.EXPLOITER_MAX_STEPS/1e6:.0f}M"
                  f"{'*' if lg.exp_gate_hit_at is not None else ''}"
                  f" wr {lg.exp_win_rate()*100:.0f}%")
    top_wr, top_id, losing = lg.pool_status()
    gap = lg.main_ep_total - lg.last_exploiter_ep
    exp_top = lg.top_exploiter_winrate()
    expstr = f"expTop {exp_top*100:.0f}%" if exp_top is not None else "expTop --"
    topstr = (f"topWR {top_wr*100:.0f}% ({top_id}) {expstr} lose {losing} "
              f"gap {gap//1000}k" if top_wr is not None
              else f"topWR -- {expstr} gap {gap//1000}k")
    return (f"step {trainer.env_steps:,} (ep {trainer.episode_count:,}) [{phase}|"
            f"snap {len(lg.snapshots)} exp {len(lg.exploiters)}] | "
            f"steps/s {sps:.0f} | "
            f"ELO {lg.current_rating:.1f} | "
            f"wr {trainer.win_rate()*100:.0f}% | {topstr} | "
            f"updates {trainer._active_agent().updates} | {loss} | "
            f"perf/cycle wait={(p['wait']-last_perf['wait'])/dc*1000:.2f} "
            f"infer={(p['infer']-last_perf['infer'])/dc*1000:.2f}ms "
            f"train={(p['train']-last_perf['train'])/(now-last_log)*100:.0f}%")


def main():
    args = build_parser().parse_args()
    apply_phase_flags(args)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    out_dir = Path(args.out)
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    # AGENT_STATE_DIM = STATE_DIM + 1: the net also sees the appended remaining-
    # hold feature. The collector/env still emit STATE_DIM; the trainer concats.
    agent = PPOAgent(C.AGENT_STATE_DIM, C.NUM_ACTIONS, device=args.device)
    coll = VecCollector(args.workers, args.envs_per_worker, args.map,
                        replay_dir=replay_dir, replay_every=args.replay_every)
    trainer = Trainer(coll, agent)
    save = CheckpointWriter(trainer, out_dir).save
    print(f"bonk2/ppo: {args.workers} workers x {args.envs_per_worker} envs "
          f"= {coll.E}, hidden {C.HIDDEN}, K={C.ACTION_REPEAT}, "
          f"gamma {C.GAMMA}, rollout {C.ROLLOUT_STEPS}, device {args.device}")

    if args.load:
        with open(args.load) as f:
            trainer.load_state(json.load(f))
    elif args.warm_actor:
        with open(args.warm_actor) as f:
            obj = json.load(f)
        # Load ONLY the actor (the trained policy). The critic is left fresh --
        # required when the critic architecture changed (e.g. scalar -> 3-atom
        # categorical, which is exactly what pipeline phase 2 does), and correct
        # in general for a reward/map change, since a value function calibrated
        # to the old objective would mislead. League and step clocks also stay
        # fresh.
        trainer.agent.actor.load_records(obj["agent"]["actor"])
        print(f"  warm-started actor from {args.warm_actor} "
              f"(fresh league, progress reset)")
        print(f"resumed: episode {trainer.episode_count}, "
              f"ELO {trainer.league.current_rating:.1f}, "
              f"snapshots {len(trainer.league.snapshots)}, "
              f"exploiters {len(trainer.league.exploiters)}")

    if args.force_exploiter:
        if trainer.league.phase == "exploiter":
            print("  --force-exploiter: already in an exploiter phase, skipping")
        else:
            force_staller = {"staller": True, "combat": False,
                             "random": None}[args.exploiter_type]
            trainer.league.start_exploiter(force_staller=force_staller)
            trainer._reset_collection()
            print(f"  --force-exploiter: forced immediately "
                 f"(type={args.exploiter_type}, "
                 f"staller={trainer.league.exp_is_staller})")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    t_start = time.perf_counter()
    last_log, last_steps, last_saved = t_start, 0, trainer.env_steps
    last_perf = dict(trainer.perf)
    last_stats = None

    # --steps is the primary stopping unit; --episodes remains as a cap.
    def running():
        if args.steps:
            return trainer.env_steps < args.steps
        return trainer.episode_count < args.episodes

    try:
        while running() and not stop["flag"]:
            trainer.cycle()
            if trainer.learner_steps() >= C.ROLLOUT_STEPS:
                last_stats, _ = trainer.run_update()

            now = time.perf_counter()
            if now - last_log >= 5.0:
                sps = (trainer.env_steps - last_steps) / (now - last_log)
                print(progress_line(trainer, last_stats, sps, last_perf,
                                    now, last_log))
                last_log, last_steps = now, trainer.env_steps
                last_perf = dict(trainer.perf)

            if trainer.env_steps - last_saved >= args.save_every:
                last_saved = trainer.env_steps
                save()
    except Exception:
        save("-crash")   # never lose progress to a bug
        raise
    finally:
        coll.stop()
    save("-final")
    print(f"done: {trainer.episode_count} episodes in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
