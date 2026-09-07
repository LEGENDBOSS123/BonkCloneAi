"""Training entry point: `python -m ppo9.train`.

Config resolution order is `default -> --phase preset -> --set overrides ->
named sugar flags`, and the RESOLVED config is echoed once (the pipeline greps
that line) and written into every checkpoint. Nothing mutates a module and
nothing edits a source file — ppo7 did both.

Ctrl-C saves a `-final` checkpoint and shuts the workers down; that is also how
the pipeline force-advances a phase.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bonkenv import VecCollector
from ppo9.agent import PPOAgent
from ppo9.config import Ppo9Config, validate, warnings_for
from ppo9.presets import apply_overrides, default, preset
from ppo9.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ppo9.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, choices=(1, 2, 3), default=3,
                   help="curriculum preset: 1 kill-idle+dense+scalar critic, "
                        "2 freeze actor+categorical critic, 3 full league")
    p.add_argument("--set", action="append", default=[], metavar="a.b=VALUE",
                   help="override any config field by dotted path (repeatable)")
    # Sugar for the flags the pipeline sets by name; all are thin wrappers over
    # --set, kept so the phase-flags line stays greppable.
    p.add_argument("--scalar-critic", action="store_true")
    p.add_argument("--dense-reward", action="store_true")
    p.add_argument("--freeze-actor", action="store_true")
    p.add_argument("--no-league", action="store_true")
    p.add_argument("--all-idle", action="store_true",
                   help="every opponent is the scripted do-nothing dummy")

    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--envs-per-worker", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--torch-threads", type=int, default=0)

    p.add_argument("--steps", type=int, default=10**11, help="env-step budget")
    p.add_argument("--save-every", type=int, default=None, help="env steps")
    p.add_argument("--out", default="../runs/ppo9", help="checkpoint directory")
    p.add_argument("--load", default=None, help="resume from a checkpoint")
    p.add_argument("--warm-actor", default=None,
                   help="load ONLY the actor from a checkpoint (phase 2 warm start)")
    p.add_argument("--replay-dir", default=None)

    p.add_argument("--force-exploiter", action="store_true",
                   help="start an exploiter phase immediately, bypassing the "
                        "mastery gate (for on-demand testing)")
    p.add_argument("--exploiter-type", choices=("random", "staller", "combat"),
                   default="random")
    p.add_argument("--dump-config", action="store_true",
                   help="print the resolved config and exit")
    return p


def resolve_config(args: argparse.Namespace) -> Ppo9Config:
    """Build the effective config from CLI arguments."""
    cfg = preset(args.phase)(default())
    cfg = apply_overrides(cfg, args.set)

    if args.scalar_critic:
        cfg = replace(cfg, critic=replace(cfg.critic, categorical=False,
                                          atoms=None, aux_gamma_coef=0.0))
    if args.dense_reward:
        cfg = replace(cfg, shaping=replace(cfg.shaping, dense_reward=True))
    if args.freeze_actor:
        cfg = replace(cfg, ppo=replace(cfg.ppo, freeze_actor=True))
    if args.all_idle:
        cfg = replace(cfg, league=replace(cfg.league, opp_idle_prob=1.0,
                                          opp_current_prob=0.0, opp_pfsp_prob=0.0,
                                          enabled=False))
    if args.no_league:
        cfg = replace(cfg, league=replace(cfg.league, enabled=False))

    run = cfg.run
    if args.workers is not None:
        run = replace(run, workers=args.workers)
    if args.envs_per_worker is not None:
        run = replace(run, envs_per_worker=args.envs_per_worker)
    if args.device is not None:
        run = replace(run, device=args.device)
    if args.save_every is not None:
        run = replace(run, save_every_steps=args.save_every)
    if args.torch_threads:
        run = replace(run, torch_threads=args.torch_threads)
    cfg = replace(cfg, run=run).resolve()
    validate(cfg)
    return cfg


def phase_flags_line(cfg: Ppo9Config) -> str:
    """The one line the pipeline greps to verify a phase started as asked."""
    return (f"  phase flags: scalar_critic={not cfg.critic.categorical} "
            f"dense={cfg.shaping.dense_reward}({cfg.shaping.dense_coef}) "
            f"all_idle={cfg.league.opp_idle_prob == 1.0} "
            f"freeze_actor={cfg.ppo.freeze_actor} league={cfg.league.enabled} "
            f"aux_gamma={cfg.critic.aux_gamma_coef}")


class CheckpointWriter:
    """Writes checkpoints and prunes old ones.

    The resolved config is written FIRST in the JSON, so a truncated file is
    still diagnosable — a truncated checkpoint has broken `--load` before.
    """

    def __init__(self, trainer: Trainer, out_dir: Path, cfg: Ppo9Config) -> None:
        self.trainer = trainer
        self.dir = out_dir
        self.cfg = cfg
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, tag: str = "") -> Path:
        tr = self.trainer
        name = (f"ppo9-step{tr.env_steps}-ep{tr.episode_count}"
                f"-elo{int(tr.league.current_rating)}{tag}.json")
        path = self.dir / name
        blob: dict[str, Any] = {"config": asdict(self.cfg)}
        blob.update(tr.serialize())
        with open(path, "w") as f:
            json.dump(blob, f)
        print(f"checkpoint saved: {path}", flush=True)
        self._prune()
        return path

    def _prune(self) -> None:
        """Keep the newest N untagged checkpoints; tagged ones are kept forever."""
        ckpts = sorted(self.dir.glob("ppo9-step*.json"),
                       key=lambda q: q.stat().st_mtime)
        regular = [q for q in ckpts if "-final" not in q.name and "-crash" not in q.name]
        for q in regular[:-self.cfg.run.checkpoint_keep]:
            q.unlink(missing_ok=True)


def progress_line(tr: Trainer, stats: dict[str, float] | None, sps: float,
                  last_perf: dict[str, float], now: float, last_log: float) -> str:
    """The single line that is the entire live view of a run.

    A watcher parses `^step N` and ` wr NN%` out of this, so keep those two
    shapes stable.
    """
    lg, p = tr.league, tr.perf
    dc = max(1, p["cycles"] - last_perf["cycles"])
    if stats:
        loss = (f"a={stats['actor_loss']:.4f} c={stats['critic_loss']:.4f} "
                f"H={stats['entropy']:.3f} "
                f"(a={stats['action_entropy']:.2f}/d={stats['duration_entropy']:.2f}) "
                f"ec={stats['ent_coef']:.3f} kl={stats['kl']:.4f} "
                f"clip={stats['clip_frac'] * 100:.0f}%"
                f"/{stats['clip_binding'] * 100:.0f}% "
                f"hold={stats.get('mean_hold', 1.0):.1f}")
        if "kl_stopped_pct" in stats:
            loss += f" klstop={stats['kl_stopped_pct']:.0f}%"
        if "decided_pct" in stats:
            loss += f" decided={stats['decided_pct']:.0f}%"
            mep = stats.get("mean_ep_decisions", float("nan"))
            if mep == mep:
                loss += f" epT={mep:.0f}"
        if "gamma_loss" in stats:
            loss += f" g2={stats['gamma_loss']:.4f}"
        if "logit_rms" in stats:
            loss += f" lrms={stats['logit_rms']:.2f}"
    else:
        loss = "warmup"

    phase = ("MAIN" if lg.phase == "main"
             else f"EXPL {lg.phase_steps / 1e6:.1f}M/"
                  f"{lg.exp_cfg.max_steps / 1e6:.0f}M"
                  f"{'*' if lg.exp_gate_hit_at is not None else ''}"
                  f" wr {lg.exp_win_rate() * 100:.0f}%")
    top_wr, top_id, losing = lg.pool_status()
    exp_top = lg.top_exploiter_winrate()
    expstr = f"expTop {exp_top * 100:.0f}%" if exp_top is not None else "expTop --"
    gap = lg.main_ep_total - lg.last_exploiter_ep
    topstr = (f"topWR {top_wr * 100:.0f}% ({top_id}) {expstr} lose {losing} "
              f"gap {gap // 1000}k" if top_wr is not None
              else f"topWR -- {expstr} gap {gap // 1000}k")

    archstr = f" arch={len(tr.archive)}" if tr.archive is not None else ""

    # Expected winrate of the matchup distribution actually being sampled.
    diag = lg.sampling_diagnostics()
    if diag is None:
        wrstr = ""
    else:
        key = "overall_wr" if lg.cfg.sampled_wr_scope == "all" else "sampled_wr"
        wrstr = f" sampleWR={diag[key] * 100:.1f}%"
        if lg.cfg.sampled_wr_debug:
            wrstr += (
                f" [tgt={diag['target'] * 100:.1f}%"
                f" eff={diag['eff_target'] * 100:.1f}%"
                f" pool={diag['sampled_wr'] * 100:.1f}%"
                f" pre={diag['base_sampled_wr'] * 100:.1f}%"
                f" wr[{diag['min_wr'] * 100:.0f}..{diag['max_wr'] * 100:.0f}]"
                f" lam={diag['lam']:+.2f}"
                f" H={diag['entropy']:.2f}/{diag['max_entropy']:.2f}"
                f" pmin={diag['min_p'] * 100:.2f}%"
                f" top=" + ",".join(f"{n}:{p * 100:.1f}%@{w * 100:.0f}"
                                    for n, p, w in diag["top"]) + "]")

    return (f"step {tr.env_steps:,} (ep {tr.episode_count:,}) [{phase}|"
            f"snap {len(lg.snapshots)} exp {len(lg.exploiters)}] | "
            f"steps/s {sps:.0f} | ELO {lg.current_rating:.1f} | "
            f"wr {tr.win_rate() * 100:.0f}% draw {tr.draw_rate() * 100:.0f}% | "
            f"{topstr}{archstr}{wrstr} | "
            f"updates {tr._active_agent().updates} | {loss} | "
            f"perf/cycle wait={(p['wait'] - last_perf['wait']) / dc * 1000:.2f} "
            f"infer={(p['infer'] - last_perf['infer']) / dc * 1000:.2f}ms "
            f"train={(p['train'] - last_perf['train']) / max(1e-9, now - last_log) * 100:.0f}%")


def main() -> None:
    args = build_parser().parse_args()
    cfg = resolve_config(args)
    if args.dump_config:
        print(cfg.to_json())
        return
    if cfg.run.torch_threads > 0:
        torch.set_num_threads(cfg.run.torch_threads)

    print(phase_flags_line(cfg), flush=True)
    for w in warnings_for(cfg):
        print(f"  warning: {w}", flush=True)

    agent = PPOAgent(cfg.agent_state_dim, cfg.num_actions, net=cfg.net,
                     ppo=cfg.ppo, critic=cfg.critic, figar=cfg.figar,
                     entropy=cfg.entropy, atoms=cfg.critic.atoms,
                     device=cfg.run.device)
    coll = VecCollector(cfg.env, cfg.run.workers, cfg.run.envs_per_worker,
                        replay_dir=args.replay_dir,
                        archive_enabled=cfg.archive.enabled,
                        archive_track_frac=cfg.archive.track_frac,
                        archive_history=cfg.archive.history,
                        archive_capacity=cfg.archive.capacity,
                        archive_spawn_prob=cfg.archive.spawn_prob,
                        archive_queue_maxsize=cfg.archive.queue_maxsize)
    trainer = Trainer(coll, agent, cfg)
    writer = CheckpointWriter(trainer, Path(args.out), cfg)

    print(f"ppo9: {cfg.run.workers} workers x {cfg.run.envs_per_worker} envs "
          f"= {trainer.E}, hidden {list(cfg.net.hidden)}, gru {cfg.net.gru_hidden}, "
          f"K={coll.k}, gamma {cfg.ppo.gamma}, rollout {cfg.ppo.rollout_steps}, "
          f"device {cfg.run.device}", flush=True)

    if args.load:
        with open(args.load) as f:
            trainer.load_state(json.load(f))
        print(f"resumed from {args.load} at step {trainer.env_steps:,}", flush=True)
    elif args.warm_actor:
        with open(args.warm_actor) as f:
            agent.actor.load_records(json.load(f)["agent"]["actor"])
        print(f"warm-started the actor from {args.warm_actor}", flush=True)

    if args.force_exploiter:
        if trainer.league.phase == "exploiter":
            print("  --force-exploiter: already in an exploiter phase, skipping")
        else:
            force = {"staller": True, "combat": False, "random": None}[args.exploiter_type]
            trainer.league.start_exploiter(force_staller=force)
            trainer._reset_collection()
            print(f"  --force-exploiter: forced immediately "
                  f"(type={args.exploiter_type})", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    stats: dict[str, float] | None = None
    last_log = t_start = time.time()
    last_steps, last_save = trainer.env_steps, trainer.env_steps
    last_perf = dict(trainer.perf)
    try:
        while trainer.env_steps < args.steps and not stop["flag"]:
            trainer.cycle()
            if trainer.learner_steps() >= cfg.ppo.rollout_steps:
                stats, _ = trainer.run_update()
            now = time.time()
            if now - last_log >= 5.0:
                sps = (trainer.env_steps - last_steps) / max(1e-9, now - last_log)
                print(progress_line(trainer, stats, sps, last_perf, now, last_log),
                      flush=True)
                last_log, last_steps = now, trainer.env_steps
                last_perf = dict(trainer.perf)
            if trainer.env_steps - last_save >= cfg.run.save_every_steps:
                writer.save()
                last_save = trainer.env_steps
    except Exception:
        writer.save("-crash")
        raise
    finally:
        coll.stop()
    writer.save("-final")
    mins = (time.time() - t_start) / 60.0
    print(f"done: {trainer.episode_count:,} episodes in {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
