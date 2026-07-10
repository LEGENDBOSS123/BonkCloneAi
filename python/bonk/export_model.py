"""Strip a training checkpoint down to a slim, deployable model.

Full checkpoints are 100+ MB — they carry the critic, 50 snapshots, and the
whole exploiter pool. Deploy (play2.mjs, play.py) only needs the actor. This
writes a ~1 MB JSON with just {algo, stateDim, agent:{actor}, progress} that
play2.mjs loads directly.

    python -m bonk.export_model runs/ppo-mp/bonk-ppo-ep9747300-elo2090-final.json \
                                --out ../models/bonk-ppo-latest.json
"""

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path or glob to a full training checkpoint")
    ap.add_argument("--out", default=str(REPO / "models/bonk-ppo-latest.json"))
    args = ap.parse_args()

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise FileNotFoundError(args.checkpoint)
    src = matches[-1]
    with open(src) as f:
        obj = json.load(f)

    actor = obj["agent"]["actor"]
    slim = {
        "version": obj.get("version", 1),
        "algo": obj.get("algo", "ppo"),
        "stateDim": obj.get("stateDim"),
        "progress": obj.get("progress", {}),
        "agent": {"actor": actor},   # actor only — no critic/snapshots/league
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slim))
    src_mb = Path(src).stat().st_size / 1e6
    out_mb = out.stat().st_size / 1e6
    print(f"{Path(src).name} ({src_mb:.0f} MB)  ->  {out} ({out_mb:.2f} MB)")


if __name__ == "__main__":
    main()
