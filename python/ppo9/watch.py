"""Tail a ppo9 training log into a compact progress file.

Imports nothing from the package on purpose: it parses the log TEXT, so it
keeps working across refactors and can watch a run started by a different
version. `train.progress_line` is the contract — `^step N` and ` wr NN%` are
the two shapes that must stay stable.

    python -m ppo9.watch ../runs/gg9-p3/train.log --out ../runs/gg9-p3/progress.txt
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

FIELDS: dict[str, str] = {
    "step": r"^step ([\d,]+)",
    "ep": r"\(ep ([\d,]+)\)",
    "sps": r"steps/s (\d+)",
    "elo": r"ELO ([\d.]+)",
    # Anchored on the leading "| " because an EXPL phase tag ALSO contains
    # " wr NN%" (the exploiter's own winrate vs the frozen main). An unanchored
    # pattern silently reports the exploiter's number as the main's.
    "wr": r"\| wr (\d+)%",
    "expwr": r"EXPL [^|]*wr (\d+)%",
    "topwr": r"topWR (\d+)%",
    "exptop": r"expTop (\d+)%",
    "snap": r"snap (\d+)",
    "exp": r"exp (\d+)",
    "upd": r"updates (\d+)",
    "a": r"a=(-?[\d.]+) c=",
    "c": r"c=([\d.]+)",
    "H": r"H=([\d.]+)",
    "Ha": r"\(a=([\d.]+)/",
    "Hd": r"/d=([\d.]+)\)",
    "ec": r"ec=([\d.]+)",
    "kl": r"kl=([\d.]+)",
    "hold": r"hold=([\d.]+)",
    "decided": r"decided=(\d+)%",
    "epT": r"epT=(\d+)",
    "g2": r"g2=([\d.]+)",
    "phase": r"\[(MAIN|EXPL[^|]*)\|",
}


def parse(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, pat in FIELDS.items():
        m = re.search(pat, line)
        if m:
            out[key] = m.group(1)
    return out


def render(rows: list[dict[str, str]], keep: int) -> str:
    """A fixed-width table of the last `keep` samples, newest last."""
    if not rows:
        return "(no progress lines yet)\n"
    cols = ["step", "ep", "phase", "sps", "elo", "wr", "expwr", "topwr", "snap", "exp",
            "H", "Ha", "Hd", "hold", "decided", "epT", "c", "kl", "g2"]
    cols = [c for c in cols if any(c in r for r in rows)]
    width = {c: max(len(c), max(len(r.get(c, "-")) for r in rows)) for c in cols}
    head = "  ".join(c.rjust(width[c]) for c in cols)
    body = "\n".join("  ".join(r.get(c, "-").rjust(width[c]) for c in cols)
                     for r in rows[-keep:])
    return f"{head}\n{body}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="training log to follow")
    ap.add_argument("--out", default=None, help="progress file (default: stdout)")
    ap.add_argument("--keep", type=int, default=40, help="rows to show")
    ap.add_argument("--every", type=float, default=15.0, help="refresh seconds")
    ap.add_argument("--once", action="store_true", help="render once and exit")
    args = ap.parse_args()

    path = Path(args.log)
    while True:
        rows: list[dict[str, str]] = []
        if path.exists():
            with open(path, errors="replace") as f:
                for line in f:
                    if line.startswith("step "):
                        rows.append(parse(line))
        text = render(rows, args.keep)
        if args.out:
            Path(args.out).write_text(text)
        else:
            print("\033[H\033[J" + text, end="", flush=True)
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
