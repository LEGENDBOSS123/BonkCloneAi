"""Strip a training checkpoint down to the deployable main actor + critic.

A ppo9 checkpoint carries the whole league — 120 snapshot actors and up to 60
exploiters — because a training run has to be able to resume the population.
None of that is needed to PLAY, and it is ~99% of the file.

    python -m ppo9.compress --load runs/ppo9/'*.json' --out models/latest.json

The output keeps the shape a checkpoint has (`agent.actor`, `agent.critic`,
`stateDim`, `config`) so `policy.load_checkpoint` / `resolve_agent("main")`,
`eval_h2h` and `play.mjs` all read it unchanged. `--actor-only` drops the
critic too, for deployment where nothing evaluates positions.

It deliberately does NOT round or quantise weights: `play.mjs` reproduces
training inference bit-for-bit, and silently changing the numbers would make a
deployment mismatch impossible to distinguish from a genuine behaviour bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .policy import load_checkpoint

# Kept for provenance and for the loaders that read them. Everything else in a
# checkpoint exists only to RESUME training.
KEEP_TOP = ("version", "algo", "savedAt", "stateDim", "progress", "config")


def compress(obj: dict, actor_only: bool = False) -> dict:
    if obj.get("algo") != "ppo9":
        raise ValueError(f"not a ppo9 checkpoint (algo={obj.get('algo')!r})")
    agent = obj["agent"]
    out: dict = {k: obj[k] for k in KEEP_TOP if k in obj}
    out["agent"] = {"actor": agent["actor"], "updates": agent.get("updates", 0)}
    if not actor_only:
        out["agent"]["critic"] = agent["critic"]
    # Explicit, so a compressed file is never mistaken for a resumable one.
    out["compressed"] = True
    out["snapshots"] = []
    out["league"] = {k: v for k, v in obj.get("league", {}).items()
                     if k != "exploiters"}
    out["league"]["exploiters"] = []
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--load", required=True, help="checkpoint path or glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--actor-only", action="store_true",
                    help="drop the critic as well")
    args = ap.parse_args()

    obj, path = load_checkpoint(args.load)
    small = compress(obj, args.actor_only)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(small, f, separators=(",", ":"))

    before = Path(path).stat().st_size
    after = out.stat().st_size
    kept = "actor" if args.actor_only else "actor + critic"
    print(f"  {path}\n    {before / 1e6:8.1f} MB  "
          f"({len(obj.get('snapshots', []))} snapshots, "
          f"{len(obj.get('league', {}).get('exploiters', []))} exploiters)")
    print(f"  {out}\n    {after / 1e6:8.1f} MB  ({kept})"
          f"   -> {after / before * 100:.2f}% of the original, "
          f"{before / max(1, after):.0f}x smaller")
    return 0


if __name__ == "__main__":
    sys.exit(main())
