"""NFSP training CLI (PyTorch). Checkpoints are rl2 browser-format JSON:
they load in train2.html ("Continue from a saved backup?") and browser saves
load here via --load. Replays land beside them — watch with replay.html.

  python -m bonk.train                       # fresh run
  python -m bonk.train --load ckpt.json      # resume (browser or python save)
  python -m bonk.train --out runs/pt1 --save-every 2000 --replay-every 500

Ctrl-C saves a final checkpoint before exiting.
"""

import argparse
import json
import signal
import time
from pathlib import Path

from . import config as C
from .env2 import LagEnv
from .nfsp import NFSPAgent
from .sim import BonkSim
from .trainer import Trainer

REPO = Path(__file__).resolve().parents[2]
NUM_ENVS = 48
TOTAL_EPISODES = 300_000


class ReplayRecorder:
    def __init__(self, out_dir: Path, every: int, trainer):
        self.dir = out_dir
        self.every = every
        self.trainer = trainer
        self.frames = []
        self.seen = 0

    def frame(self, env, res):
        p0, p1 = env.sim.players
        a0, a1 = env.applied_actions()
        r3 = lambda v: round(v, 3)
        self.frames.append([
            r3(p0.pos[0]), r3(p0.pos[1]), 0,
            r3(p1.pos[0]), r3(p1.pos[1]), 0,
            a0, a1,
        ])
        if not res["done"]:
            return
        self.seen += 1
        if self.seen % self.every == 0:
            result = ("draw" if res["timeout"] or all(res["dead"])
                      else "p1-wins" if res["dead"][1] else "p2-wins")
            path = self.dir / f"ep{self.trainer.episode_count + 1}-{result}.json"
            path.write_text(json.dumps({
                "version": 1, "tps": C.TPS, "actionRepeat": C.ACTION_REPEAT,
                "inputLag": C.INPUT_LAG, "result": result, "frames": self.frames,
            }))
            print(f"replay saved: {path} ({len(self.frames)} frames, {result})")
        self.frames = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"))
    ap.add_argument("--load", help="checkpoint JSON (python or browser save)")
    ap.add_argument("--out", default=str(REPO / "runs/pytorch"))
    ap.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--replay-every", type=int, default=500)
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--learn-start", type=int, help="override warmup (tests)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    with open(args.map) as f:
        map_json = json.load(f)
    envs = [LagEnv(BonkSim(map_json)) for _ in range(args.num_envs)]
    agent = NFSPAgent(envs[0].state_dim, C.NUM_ACTIONS, device=args.device)
    trainer = Trainer(envs, agent)
    if args.learn_start is not None:
        import bonk.trainer as trainer_mod
        trainer_mod.LEARN_START = args.learn_start
    trainer.recorder = ReplayRecorder(replay_dir, args.replay_every, trainer)
    print(f"NFSP/pytorch: {args.num_envs} envs, obs {envs[0].state_dim}, "
          f"device {args.device}, lag {C.INPUT_LAG}")

    if args.load:
        with open(args.load) as f:
            trainer.load_state(json.load(f))
        print(f"resumed: episode {trainer.episode_count}, "
              f"ELO {trainer.current_rating:.1f}, snapshots {len(trainer.snapshots)}")

    def save_checkpoint(tag=""):
        name = (f"bonk-nfsp-ep{trainer.episode_count}"
                f"-elo{round(trainer.current_rating)}{tag}.json")
        path = out_dir / name
        path.write_text(json.dumps(trainer.serialize()))
        print(f"checkpoint saved: {path}")
        return path

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    t_start = time.perf_counter()
    last_log = t_start
    last_steps = 0
    last_perf = dict(trainer.perf)
    last_saved = trainer.episode_count

    while trainer.episode_count < args.episodes and not stop["flag"]:
        trainer.tick()
        trainer.train_tick()

        now = time.perf_counter()
        if now - last_log >= 5.0:
            p = trainer.perf
            dt = max(1, p["ticks"] - last_perf["ticks"])
            sps = (trainer.env_steps - last_steps) / (now - last_log)
            s = agent.stats
            print(
                f"ep {trainer.episode_count} | steps/s {sps:.0f} | "
                f"ELO {trainer.current_rating:.1f} (wr {trainer.eval_win_rate()*100:.0f}%) | "
                f"BRvAVG {trainer.br_win_rate()*100:.1f}% | "
                f"replay {trainer.replay.size} | sac {agent.sac_steps} | "
                f"H {s['entropy']:.3f} a {s['alpha']:.4f} avgCE {s['avg_loss']:.4f} | "
                f"perf decide={(p['decide']-last_perf['decide'])/dt*1000:.2f} "
                f"phys={(p['phys']-last_perf['phys'])/dt*1000:.2f} "
                f"train={(p['train']-last_perf['train'])/dt*1000:.2f}ms/tick"
            )
            last_log, last_steps, last_perf = now, trainer.env_steps, dict(p)

        if trainer.episode_count - last_saved >= args.save_every:
            last_saved = trainer.episode_count
            save_checkpoint()

    save_checkpoint("-final")
    print(f"done: {trainer.episode_count} episodes in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
