"""Tail a training log into a compact, human-readable progress file.

    python -m bonk4.watch /tmp/gg.log --out ../runs/ppo5gg/progress.txt

Deliberately a SEPARATE PROCESS rather than an extra write inside train.py:
the trainer is the thing under test, and a formatting change should never be
able to take a multi-hour run down. It reads the log the trainer already
writes, so it can be started, stopped or restarted at any point without
touching training.

Open the output file in an editor that reloads on change and it updates live.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from pathlib import Path

# Pulled out of the trainer's single-line-per-update format.
PAT = {
    "ep":    r"^ep (\d+)",
    "upd":   r"updates (\d+)",
    "wr":    r"wr (\d+)%",
    "elo":   r"ELO ([\d.]+)",
    "H":     r"H=([\d.]+)",
    "ec":    r"ec=([\d.]+)",
    "ev":    r"ev=(-?[\d.]+)",
    "oacc":  r"oacc=(\d+)%",
    "kl":    r"kl=([\d.]+)",
    "sp":    r"spawn=(\d+)%",
    "sps":   r"steps/s (\d+)",
    "T":     r"T=(\d+)",
    "lab":   r"lab=(\d+)%",
    "snap":  r"snap (\d+)",
    "exp":   r"exp (\d+)",
    "hca":   r"hca=(\d+)%",
}


def parse(line: str):
    if "updates " not in line:
        return None
    row = {}
    for k, p in PAT.items():
        m = re.search(p, line)
        row[k] = m.group(1) if m else None
    return row if row["upd"] else None


def spark(vals, lo=None, hi=None, width=40):
    """Unicode sparkline over the most recent `width` values."""
    v = [x for x in vals if x is not None][-width:]
    if len(v) < 2:
        return ""
    lo = min(v) if lo is None else lo
    hi = max(v) if hi is None else hi
    if hi - lo < 1e-9:
        return "─" * len(v)
    bars = "▁▂▃▄▅▆▇█"
    # Clamped at BOTH ends: values outside [lo, hi] are normal here (ev is
    # hugely negative during warmup, when Var(outcome) ~ 0 makes the ratio
    # explode), and a negative index silently walks off the string.
    return "".join(bars[max(0, min(7, int((x - lo) / (hi - lo) * 7.999)))]
                   for x in v)


def fmt(rows, log_path, t0, phase):
    r = rows[-1]
    f = lambda k, d=0.0: float(r[k]) if r[k] is not None else d           # noqa: E731
    hist = lambda k: [float(x[k]) for x in rows if x[k] is not None]      # noqa: E731

    el = time.time() - t0
    out = []
    out.append(f"  {phase}")
    out.append(f"  log {log_path}")
    out.append(f"  updated {time.strftime('%H:%M:%S')}   "
               f"running {int(el)//3600}h{int(el)%3600//60:02d}m")
    out.append("")
    out.append(f"  episodes {int(f('ep')):>9,}      updates {int(f('upd')):>6,}"
               f"      steps/s {int(f('sps')):>6,}")
    out.append(f"  league    snapshots {r['snap'] or '-':>3}   "
               f"exploiters {r['exp'] or '-':>3}      spawn {r['sp'] or '-'}%")
    out.append("")

    # (label, key, low, high, format) -- ranges fixed so the bars are
    # comparable across time rather than autoscaling to the recent window.
    rng = [
        ("win rate   ", "wr",   0.0, 100.0, "{:.0f}%"),
        ("outcome acc", "oacc", 33.0, 100.0, "{:.0f}%"),
        ("expl. var  ", "ev",   0.0, 1.0,   "{:+.3f}"),
        ("entropy H  ", "H",    0.0, 2.89,  "{:.3f}"),
        ("kl         ", "kl",   0.0, 0.03,  "{:.4f}"),
        ("ELO        ", "elo",  None, None, "{:.0f}"),
    ]
    if r["hca"] is not None:
        rng.insert(3, ("hindsight  ", "hca", 33.0, 100.0, "{:.0f}%"))

    for label, key, lo, hi, ff in rng:
        h = hist(key)
        if not h:
            continue
        out.append(f"  {label} {ff.format(h[-1]):>8}  {spark(h, lo, hi)}")

    out.append("")
    out.append(f"  H is at {f('H'):.3f} of a {2.890:.3f} uniform maximum "
               f"({f('H') / 2.890 * 100:.1f}%) -- the policy is")
    out.append("  only meaningfully committed once this falls well below it.")
    out.append("")
    out.append(f"  data health   labelled {r['lab'] or '-'}%   "
               f"mean episode len {r['T'] or '-'} decisions")
    out.append("")
    out.append("  last 12 updates")
    out.append(f"  {'upd':>7} {'episodes':>10} {'wr':>5} {'oacc':>6} "
               f"{'ev':>8} {'H':>7} {'kl':>8}")
    for x in rows[-12:]:
        g = lambda k, d="-": x[k] if x[k] is not None else d              # noqa: E731
        out.append(f"  {g('upd'):>7} {int(g('ep', 0)):>10,} {g('wr'):>4}% "
                   f"{g('oacc'):>5}% {g('ev'):>8} {g('H'):>7} {g('kl'):>8}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None, help="default: <log dir>/progress.txt")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--phase", default="training", help="label for this run")
    args = ap.parse_args()

    out = Path(args.out or (Path(args.log).parent / "progress.txt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, t0, pos = deque(maxlen=4000), time.time(), 0

    print(f"  watching {args.log}\n  writing  {out}\n  Ctrl-C to stop")
    while True:
        try:
            if os.path.exists(args.log):
                with open(args.log, errors="replace") as f:
                    f.seek(pos)
                    for line in f:
                        r = parse(line)
                        if r:
                            rows.append(r)
                    pos = f.tell()
            if rows:
                tmp = out.with_suffix(".tmp")
                # Written via a temp file + rename so a reader never catches a
                # half-written file mid-update.
                tmp.write_text(fmt(list(rows), args.log, t0, args.phase))
                tmp.replace(out)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  stopped")
            return


if __name__ == "__main__":
    main()
