"""Diagnostic: how much is the FiGAR duration head actually doing, right now?

Loads an actor checkpoint and self-plays it against itself on the real map,
replaying the EXACT FiGAR/hidden-state bookkeeping the trainer uses
(`figar.HoldTracker`, per-seat minGRU hidden, single-frame recurrent obs), and
records the duration head's own probability distribution at every FREE
decision (never at a held/forced-repeat row -- those aren't real samples).

Reports three things a training-curve log line (`hold=`) can't show:
  1. The marginal distribution over the 7 duration buckets.
  2. Duration-head ENTROPY -- separated from the combined action+duration H the
     training log reports, so "is the head collapsed" isn't confounded by the
     action head's entropy.
  3. Whether the chosen duration is STATE-DEPENDENT: split decisions by
     distance-to-opponent (the obvious reactive-vs-boring proxy already in the
     observation, so this adds no bonk-specific human theory) and compare mean
     duration + duration entropy between the near and far halves. A head that
     is doing its job holds longer when far / less when close; a head that
     picked one duration and ignores the state won't show a gap here even if
     its marginal mean "looks" nontrivial.

Usage:
    python -m ppo7.analyze_duration_head --ckpt <path> [--envs 64] [--cycles 1500]
"""
import argparse
import json

import numpy as np
import torch

from bonk3.simadapter import make_sim

from . import config as C
from .env import LagEnv
from .figar import HoldTracker
from .mingru import MinGRUNet
from .policy import resolve_agent


def analyze(ckpt_path, spec="main", n_envs=64, n_cycles=1500, seed=0):
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    weights, label = resolve_agent(ckpt, spec)
    recs = weights.get("records", weights) if isinstance(weights, dict) else weights
    if not (isinstance(recs, dict) and recs.get("type") == "mingru"):
        raise ValueError("this script assumes a recurrent (minGRU) actor; "
                         f"got type={recs.get('type') if isinstance(recs, dict) else 'flat-mlp'}")
    net = MinGRUNet.from_records(recs)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    E, H = n_envs, net.H
    envs = [LagEnv(make_sim(C.MAP_NAME, engine=C.ENGINE)) for _ in range(E)]
    for e in envs:
        e.reset()
    hidden = [torch.zeros(E, H), torch.zeros(E, H)]     # per seat
    holds = [HoldTracker(E), HoldTracker(E)]

    rows = []   # (dist_m, dur_idx, dur_entropy, action_entropy)
    for cyc in range(n_cycles):
        applied = np.zeros((E, 2), dtype=np.int64)
        for seat in (0, 1):
            raw = np.stack([e.decision_state(seat) for e in envs]).astype(np.float32)
            hf = holds[seat].feature()
            x = torch.from_numpy(np.concatenate([raw, hf], axis=1))
            with torch.no_grad():
                logits, hidden[seat] = net.step(x, hidden[seat])
            la, ld = logits[:, :C.NUM_ACTIONS], logits[:, C.NUM_ACTIONS:]
            pa, pd = torch.softmax(la, 1), torch.softmax(ld, 1)
            ent_a = (-(pa * (pa + 1e-9).log()).sum(1)).numpy()
            ent_d = (-(pd * (pd + 1e-9).log()).sum(1)).numpy()
            a = torch.multinomial(pa, 1).squeeze(1).numpy()
            d = torch.multinomial(pd, 1).squeeze(1).numpy()
            free = holds[seat].free_mask()
            if free.any():
                # rel dx,dy live at raw[24],raw[25], POS_SCALEd -> back to metres.
                dist = np.hypot(raw[:, 24], raw[:, 25]) / C.POS_SCALE
                for i in np.nonzero(free)[0]:
                    rows.append((float(dist[i]), int(d[i]),
                                float(ent_d[i]), float(ent_a[i])))
            fi = np.nonzero(free)[0]
            holds[seat].latch(fi, a[free], d[free])
            applied[:, seat] = holds[seat].advance()

        for i, e in enumerate(envs):
            e.set_decision(0, int(applied[i, 0]))
            e.set_decision(1, int(applied[i, 1]))
        ended = np.zeros(E, dtype=bool)
        for _ in range(C.ACTION_REPEAT):
            for i, e in enumerate(envs):
                if ended[i]:
                    continue
                res = e.tick()
                if res["done"]:
                    ended[i] = True
        if ended.any():
            for i in np.nonzero(ended)[0]:
                envs[i].reset()
            for seat in (0, 1):
                holds[seat].reset(ended)
                hidden[seat][ended] = 0.0

    dist, dur, ent_d, ent_a = map(np.asarray, zip(*rows))
    return dict(label=label, n=len(rows), dist=dist, dur=dur,
               ent_d=ent_d, ent_a=ent_a)


def report(res):
    n = res["n"]
    durs = np.asarray(C.DURATIONS)
    cycles_sampled = durs[res["dur"]]
    print(f"agent: {res['label']}   free decisions sampled: {n}")
    print()
    print("marginal duration distribution:")
    counts = np.bincount(res["dur"], minlength=C.NUM_DURATIONS)
    for d, c in zip(durs, counts):
        bar = "#" * int(60 * c / n)
        print(f"  hold={d:>3} cyc  {c/n*100:5.1f}%  {bar}")
    print(f"  mean hold: {cycles_sampled.mean():.2f} cycles"
         f"  (median {np.median(cycles_sampled):.0f}, log(7)=uniform would be"
         f" {np.log(C.NUM_DURATIONS):.3f} nats)")
    print()
    print(f"duration-head entropy: mean {res['ent_d'].mean():.3f} nats"
         f"  (0=deterministic, {np.log(C.NUM_DURATIONS):.3f}=uniform over 7)")
    print(f"action-head entropy:   mean {res['ent_a'].mean():.3f} nats"
         f"  ({np.log(C.NUM_ACTIONS):.3f}=uniform over 18)")
    print()
    med = np.median(res["dist"])
    near, far = res["dist"] < med, res["dist"] >= med
    print(f"state-dependence check (median distance {med:.1f}m splits the data):")
    print(f"  NEAR ({near.sum()} rows): mean hold "
         f"{cycles_sampled[near].mean():.2f} cyc, dur-entropy {res['ent_d'][near].mean():.3f}")
    print(f"  FAR  ({far.sum()} rows): mean hold "
         f"{cycles_sampled[far].mean():.2f} cyc, dur-entropy {res['ent_d'][far].mean():.3f}")
    gap = cycles_sampled[far].mean() - cycles_sampled[near].mean()
    print(f"  far-minus-near hold gap: {gap:+.2f} cycles"
         f"  ({'state-dependent, holds LONGER when far (design intent)' if gap > 0.5 else 'no meaningful gap -- duration looks state-INDEPENDENT' if abs(gap) <= 0.5 else 'holds LONGER when near (unexpected)'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", default="main")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--cycles", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report(analyze(args.ckpt, args.spec, args.envs, args.cycles, args.seed))
