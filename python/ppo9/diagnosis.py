"""Self-play diagnosis: record N games of the main against itself, then analyse.

The training log reports what the agent SCORED. This reports what it DID, and
how well the critic understood it. Two passes, so an expensive recording can be
re-analysed without replaying anything::

    python -m ppo9.diagnosis record --load runs/ppo9/'*.json' --games 100
    python -m ppo9.diagnosis analyze --data runs/diagnosis.npz

Self-play is what makes most of this free: both seats run the SAME policy, so
the matchup is exactly symmetric, the true win probability is 0.5 by
construction, and every prediction is checkable against an observed outcome.

The four analyses, and the decision each one drives:

1. **Death attribution** — self-death vs contested death. `bonkenv` exposes no
   contact flag, so the proxy is the minimum inter-disc distance over the
   decisions preceding a death: if the discs never closed to within
   `contact_m`, nothing killed it but the map. CLAUDE.md's first open problem
   is that the agent "rushes and falls off the edge when the opponent dodges",
   and there is currently no number for it.
   -> is the juke problem still live, i.e. is a frame-stack retrain warranted?

2. **Critic calibration** — a reliability diagram over predicted P(win). Every
   advantage, GAE, and the value-swing archive ride on this critic; if it is
   miscalibrated the archive is detecting noise rather than fumbles. Self-play
   also pins two exact checks: mean predicted value must be ~0 and the two
   seats must win equally.
   -> is the critic trustworthy at all?

3. **Value-swing distribution** — the empirical size of the recovery/fumble
   swings `ppo9.archive` hunts for.
   -> turns `archive.value_threshold` from a guess into a percentile.

4. **Action / duration occupancy** — marginal probability per action and per
   FiGAR duration, plus how peaked the policy is. `EntropyConfig` documents
   this as an UNRECOVERABLE failure: "once a discrete action's probability
   truly reaches 0 under on-policy sampling PPO gets no gradient for it and it
   cannot recover", and ppo7 lost its 32/64 duration buckets exactly that way.
   -> does `duration_coef` need raising, or should dead buckets be dropped?

Logit RMS is reported alongside (4) because it costs nothing here and has a
known failure mode: a saturated softmax makes the entropy bonus's gradient
vanish, so logits can grow without bound and every step becomes a KL jump.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from bonk3.simadapter import make_sim

from bonkenv import (OPP_OFF, SELF_OFF, EnvConfig, LagEnv, Outcome,
                     build_layout)
from .config import FigarConfig
from .mingru import MinGRUNet
from .policy import load_checkpoint, load_policy, resolve_agent

# Per-decision columns in the recorded table. Kept to scalars on purpose: a
# 100-game run is ~70k decisions, and storing the full 129-float observation
# for each would be ~35MB per seat for data no analysis below reads.
COLS = ["game", "seat", "t", "x", "y", "dist", "value", "p_win", "p_draw",
        "p_loss", "action", "duration", "entropy", "logit_rms", "logit_max",
        "outcome", "died", "timeout"]
CI = {c: i for i, c in enumerate(COLS)}


class _Scored:
    """An `ActorPolicy` that also reports what the nets were thinking.

    Deliberately re-implements `ActorPolicy.act`'s FiGAR bookkeeping rather
    than wrapping it: the hold gates the ACTION but the memory steps every
    cycle, and a diagnosis that got that wrong would silently measure a
    different policy than the one that trained.
    """

    def __init__(self, actor_records, critic_records, figar: FigarConfig,
                 atoms: np.ndarray) -> None:
        self.pol = load_policy(actor_records, figar)
        self.critic = MinGRUNet.from_records(critic_records)
        self.critic.eval()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        self.atoms = torch.tensor(atoms, dtype=torch.float32)
        self.hc: torch.Tensor | None = None

    def reset(self, n: int) -> None:
        self.pol.reset(n)
        self.hc = self.critic.zero_h(n)

    def reset_env(self, i: int) -> None:
        self.pol.reset_env(i)
        if self.hc is not None:
            self.hc[i] = 0.0

    @torch.no_grad()
    def step(self, states: np.ndarray):
        """-> (applied_action, sampled_duration, probs[3], entropy, logits stats)."""
        p = self.pol
        x = np.ascontiguousarray(states, dtype=np.float32)
        n = x.shape[0]
        if p.hold is None or len(p.hold) != n:
            self.reset(n)
        hf = (p.hold.astype(np.float32) / p._maxdur)[:, None]
        inp = torch.from_numpy(np.concatenate([x, hf], 1)).float()

        logits, p.h = p.net.step(inp, p.h)
        la, ld = logits[:, :p.num_actions], logits[:, p.num_actions:]
        pa = torch.softmax(la, 1)
        a = torch.multinomial(pa, 1).squeeze(1).numpy()
        d = torch.multinomial(torch.softmax(ld, 1), 1).squeeze(1).numpy()
        ent = -(pa * torch.log(pa.clamp_min(1e-9))).sum(1).numpy()

        c_out, self.hc = self.critic.step(inp, self.hc)
        probs = torch.softmax(c_out, dim=-1)
        value = (probs @ self.atoms).numpy()

        free = p.hold == 0
        p.held_a[free] = a[free]
        p.hold[free] = p._durs[d[free]]
        applied = p.held_a.copy()
        p.hold -= 1
        return (applied, d, probs.numpy(), value, ent,
                logits.pow(2).mean(1).sqrt().numpy(),
                logits.abs().amax(1).numpy())


def record(args) -> int:
    ckpt, path = load_checkpoint(args.load)
    records, label = resolve_agent(ckpt, "main")
    critic = ckpt["agent"]["critic"]
    cfg = EnvConfig().resolve()
    layout = build_layout(cfg.obs)
    figar = FigarConfig()
    # The atoms the checkpoint's critic was trained under. `resolve()` derives
    # them from the env's reward values, which is where they came from.
    from .presets import default
    atoms = np.asarray(default().resolve().critic.atoms, dtype=np.float32)
    print(f"  {path}  ({label}, atoms={tuple(atoms)})")

    # BOTH seats are the same weights but need INDEPENDENT hidden state, so
    # they are separate objects. Sharing one would silently couple the seats.
    seats = [_Scored(records, critic, figar, atoms) for _ in range(2)]
    n_envs = min(args.envs, args.games)
    envs = [LagEnv(make_sim(cfg.engine.map_name, engine=cfg.engine.engine),
                   cfg, layout) for _ in range(n_envs)]
    for e in envs:
        e.reset()
    for s in seats:
        s.reset(n_envs)

    pos_scale = cfg.obs.pos_scale
    rows: list[np.ndarray] = []
    # One open block per (env, seat); flushed with its outcome when the episode
    # ends, since outcome/died/timeout are only known then.
    open_rows: list[list[list[float]]] = [[[], []] for _ in range(n_envs)]
    step_i = np.zeros(n_envs, dtype=np.int64)
    game_id = np.arange(n_envs, dtype=np.int64)
    next_game = n_envs
    finished = 0
    tick = 0

    while finished < args.games:
        if tick % cfg.episode.action_repeat == 0:
            obs = [np.stack([e.decision_state(s) for e in envs]) for s in (0, 1)]
            out = [seats[s].step(obs[s]) for s in (0, 1)]
            for s in (0, 1):
                applied, dur, probs, value, ent, lrms, lmax = out[s]
                o = obs[s]
                sx = o[:, SELF_OFF] / pos_scale
                sy = o[:, SELF_OFF + 1] / pos_scale
                ox = o[:, OPP_OFF] / pos_scale
                oy = o[:, OPP_OFF + 1] / pos_scale
                dist = np.hypot(ox - sx, oy - sy)
                for i, e in enumerate(envs):
                    e.set_decision(s, int(applied[i]))
                    open_rows[i][s].append([
                        game_id[i], s, step_i[i], sx[i], sy[i], dist[i],
                        value[i], probs[i, int(Outcome.WIN)],
                        probs[i, int(Outcome.DRAW)], probs[i, int(Outcome.LOSS)],
                        applied[i], dur[i], ent[i], lrms[i], lmax[i],
                        -1.0, 0.0, 0.0])
            step_i += 1

        for i, e in enumerate(envs):
            res = e.tick()
            if not res.done:
                continue
            for s in (0, 1):
                blk = open_rows[i][s]
                if blk:
                    arr = np.asarray(blk, dtype=np.float32)
                    arr[:, CI["outcome"]] = float(int(res.outcome[s]))
                    arr[:, CI["died"]] = 1.0 if res.dead[s] else 0.0
                    arr[:, CI["timeout"]] = 1.0 if res.timeout else 0.0
                    rows.append(arr)
                open_rows[i][s] = []
            finished += 1
            if finished % max(1, args.games // 10) == 0:
                print(f"    {finished}/{args.games} games", flush=True)
            e.reset()
            for s in seats:
                s.reset_env(i)
            step_i[i] = 0
            game_id[i] = next_game
            next_game += 1
        tick += 1

    table = np.concatenate(rows, 0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, table=table, cols=np.array(COLS),
                        atoms=atoms, durations=np.asarray(figar.durations),
                        games=args.games, ckpt=str(path))
    print(f"\n  {len(table):,} decisions from {args.games} games "
          f"-> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


# ── analyses ───────────────────────────────────────────────────────────────
def _episodes(t: np.ndarray):
    """Yield each (game, seat) block in recorded order."""
    key = t[:, CI["game"]] * 2 + t[:, CI["seat"]]
    cuts = np.flatnonzero(np.diff(key)) + 1
    return np.split(t, cuts)


def death_attribution(t: np.ndarray, contact_m: float, look: int) -> None:
    print("\n1. DEATH ATTRIBUTION  (self-death = never closed to "
          f"{contact_m:.1f}m in the last {look} decisions)")
    self_d = cont_d = 0
    closing = 0
    for ep in _episodes(t):
        if ep[-1, CI["died"]] < 0.5:
            continue
        tail = ep[-look:]
        near = float(tail[:, CI["dist"]].min())
        if near > contact_m:
            self_d += 1
            if len(tail) > 4 and tail[-1, CI["dist"]] < tail[0, CI["dist"]]:
                closing += 1
        else:
            cont_d += 1
    total = self_d + cont_d
    if not total:
        print("   no deaths recorded")
        return
    print(f"   deaths {total:,}:  SELF {self_d:,} ({self_d / total * 100:.1f}%)"
          f"   contested {cont_d:,} ({cont_d / total * 100:.1f}%)")
    if self_d:
        print(f"   of the self-deaths, {closing / self_d * 100:.1f}% happened while "
              "CLOSING on the opponent (the juke signature)")


def critic_calibration(t: np.ndarray) -> None:
    print("\n2. CRITIC CALIBRATION")
    ends = np.array([ep[-1] for ep in _episodes(t)])
    won = (ends[:, CI["outcome"]] == int(Outcome.WIN)).mean()
    print(f"   seat-0 win rate {(ends[ends[:, CI['seat']] == 0][:, CI['outcome']] == int(Outcome.WIN)).mean() * 100:5.1f}%"
          f"   seat-1 {(ends[ends[:, CI['seat']] == 1][:, CI['outcome']] == int(Outcome.WIN)).mean() * 100:5.1f}%"
          "   (self-play: must be equal)")
    print(f"   mean predicted value {t[:, CI['value']].mean():+.4f}"
          "   (self-play: must be ~0)")

    y = (t[:, CI["outcome"]] == int(Outcome.WIN)).astype(np.float64)
    p = t[:, CI["p_win"]].astype(np.float64)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    print("   reliability (predicted P(win) -> realised):")
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() < 20:
            continue
        pred, real, n = p[m].mean(), y[m].mean(), int(m.sum())
        ece += (n / len(p)) * abs(pred - real)
        bar = "#" * int(round(real * 30))
        print(f"     [{edges[b]:.1f},{edges[b+1]:.1f})  n={n:>7,}  "
              f"pred {pred * 100:5.1f}%  real {real * 100:5.1f}%  {bar}")
    print(f"   expected calibration error {ece * 100:.2f}pt "
          "(<2 good, >10 the critic is not measuring what it claims)")


def value_swings(t: np.ndarray, look: int) -> None:
    print(f"\n3. VALUE SWINGS  (over a {look}-decision window)")
    rec, fum = [], []
    for ep in _episodes(t):
        v = ep[:, CI["value"]]
        if len(v) < look + 2:
            continue
        # Terminal-adjacent rows are excluded for the same reason
        # `ppo9.archive` excludes them: a value racing to +-1 before a real
        # win/loss is the critic reading the outcome, not a mid-game swing.
        v = v[:-5] if len(v) > look + 7 else v
        for i in range(look, len(v)):
            w = v[i - look:i + 1]
            rec.append(float(w[-1] - w.min()))
            fum.append(float(w.max() - w[-1]))
    if not rec:
        print("   episodes too short to measure")
        return
    for name, arr in (("recovery (low->high)", np.array(rec)),
                      ("fumble  (high->low)", np.array(fum))):
        q = np.percentile(arr, [50, 90, 95, 99])
        print(f"   {name}: mean {arr.mean():.3f}  p50 {q[0]:.3f}  p90 {q[1]:.3f}"
              f"  p95 {q[2]:.3f}  p99 {q[3]:.3f}  max {arr.max():.3f}")
    both = np.maximum(np.array(rec), np.array(fum))
    for thr in (0.1, 0.2, 0.3, 0.5, 0.6):
        print(f"     archive.value_threshold={thr:.2f} would fire on "
              f"{(both > thr).mean() * 100:6.2f}% of windows")


def action_occupancy(t: np.ndarray, durations: np.ndarray) -> None:
    print("\n4. ACTION / DURATION OCCUPANCY")
    a = t[:, CI["action"]].astype(int)
    counts = np.bincount(a, minlength=18) / len(a)
    dead = np.flatnonzero(counts == 0)
    print("   applied-action marginals (18 joint actions):")
    print("     " + " ".join(f"{c * 100:4.1f}" for c in counts))
    print(f"   never applied: {list(dead) if len(dead) else 'none'}")

    d = t[:, CI["duration"]].astype(int)
    dc = np.bincount(d, minlength=len(durations)) / len(d)
    print("   FiGAR duration bucket usage:")
    for k, (dur, frac) in enumerate(zip(durations, dc)):
        flag = "  <-- DEAD" if frac == 0 else ("  <-- near-dead" if frac < 0.005 else "")
        print(f"     {dur:>3} cycles: {frac * 100:6.2f}%{flag}")
    print(f"   mean sampled duration {float(durations[d].mean()):.2f} cycles")

    ent = t[:, CI["entropy"]]
    print(f"   action entropy: mean {ent.mean():.3f} of max {np.log(18):.3f} "
          f"({ent.mean() / np.log(18) * 100:.0f}% of uniform), "
          f"p05 {np.percentile(ent, 5):.3f} p95 {np.percentile(ent, 95):.3f}")
    print(f"   logit RMS: mean {t[:, CI['logit_rms']].mean():.2f}  "
          f"p99 {np.percentile(t[:, CI['logit_rms']], 99):.2f}  "
          f"max|logit| {t[:, CI['logit_max']].max():.2f}")
    print("     (~3 healthy; a saturated softmax kills the entropy gradient, "
          "so logits can grow unbounded and every step becomes a KL jump)")


def analyze(args) -> int:
    z = np.load(args.data, allow_pickle=False)
    t, durs = z["table"], z["durations"]
    print(f"  {args.data}: {len(t):,} decisions, {int(z['games'])} games, "
          f"{str(z['ckpt'])}")
    ep_lens = [len(e) for e in _episodes(t)]
    print(f"  episode length: mean {np.mean(ep_lens):.0f} decisions, "
          f"median {np.median(ep_lens):.0f}, max {max(ep_lens)}")
    ends = np.array([e[-1] for e in _episodes(t)])
    n_draw = float((ends[:, CI["outcome"]] == int(Outcome.DRAW)).mean())
    print(f"  draws {n_draw * 100:.1f}% of episode-ends "
          f"(timeout-flagged {ends[:, CI['timeout']].mean() * 100:.1f}%)")
    death_attribution(t, args.contact_m, args.look)
    critic_calibration(t)
    value_swings(t, args.swing_look)
    action_occupancy(t, durs)
    return 0


def arena(args) -> int:
    """Greedy vs sampled, three ways — does sampling FIND wins or MAKE mistakes?

    The entropy bonus buys exploration during training. At deploy time the same
    stochasticity is either still earning its keep (sampling finds lines argmax
    misses) or it is noise on a policy that already knows what to do. The three
    conditions separate those:

      G vs G   both deterministic. Symmetric, so ~50/50; its real job is to
               show the draw rate and episode length of pure argmax play.
      G vs S   the decisive one. If greedy wins clearly, each sampled deviation
               is on net a MISTAKE.
      S vs S   the training distribution, as a baseline.

    Wins are attributed to the SEAT and sides are NOT swapped, so the table
    reads exactly as run. That leaves seat bias in the numbers — the two seats
    are not perfectly symmetric — so row 2 is read against row 1 and row 3,
    which carry the same bias.
    """
    ckpt, path = load_checkpoint(args.load)
    records, label = resolve_agent(ckpt, "main")
    cfg = EnvConfig().resolve()
    layout = build_layout(cfg.obs)
    figar = FigarConfig()
    print(f"  {path}  ({label})\n")

    def play(greedy0: bool, greedy1: bool, games: int):
        n = min(args.envs, games)
        p0, p1 = load_policy(records, figar), load_policy(records, figar)
        p0.reset(n)
        p1.reset(n)
        envs = [LagEnv(make_sim(cfg.engine.map_name, engine=cfg.engine.engine),
                       cfg, layout) for _ in range(n)]
        for e in envs:
            e.reset()
        w0 = w1 = draws = done_n = 0
        lens: list[int] = []
        step = np.zeros(n, dtype=np.int64)
        tick = 0
        while done_n < games:
            if tick % cfg.episode.action_repeat == 0:
                a0 = p0.act(np.stack([e.decision_state(0) for e in envs]), greedy0)
                a1 = p1.act(np.stack([e.decision_state(1) for e in envs]), greedy1)
                for i, e in enumerate(envs):
                    e.set_decision(0, int(a0[i]))
                    e.set_decision(1, int(a1[i]))
                step += 1
            for i, e in enumerate(envs):
                res = e.tick()
                if not res.done:
                    continue
                if res.outcome[0] == Outcome.WIN:
                    w0 += 1
                elif res.outcome[0] == Outcome.LOSS:
                    w1 += 1
                else:
                    draws += 1
                lens.append(int(step[i]))
                done_n += 1
                step[i] = 0
                e.reset()
                p0.reset_env(i)
                p1.reset_env(i)
            tick += 1
        return w0, w1, draws, done_n, float(np.mean(lens))

    print(f"  {args.games} games per row\n")
    print(f"  {'AI 1 isGreedy':>13} {'AI 2 isGreedy':>13} {'AI 1 wins':>10} "
          f"{'AI 2 wins':>10} {'Draws':>7} | {'draw%':>6} {'mean len':>9}")
    print("  " + "-" * 80)
    out = {}
    for g0, g1 in ((True, True), (True, False), (False, False)):
        w0, w1, d, n, ln = play(g0, g1, args.games)
        out[(g0, g1)] = (w0, w1, d, n)
        print(f"  {str(g0):>13} {str(g1):>13} {w0:>10} {w1:>10} {d:>7} | "
              f"{d / n * 100:>5.1f}% {ln:>9.0f}", flush=True)

    w0, w1, d, n = out[(True, False)]
    dec = w0 + w1
    if dec:
        wr = w0 / dec
        se = (wr * (1 - wr) / dec) ** 0.5
        b0, b1, _, _ = out[(True, True)]
        base = b0 / max(1, b0 + b1)
        print(f"\n  row 2 decisive only: the GREEDY side (AI 1) took "
              f"{wr * 100:.1f}% +-{se * 196:.1f} (95% CI, n={dec})")
        print(f"  row 1 gives the seat-0 baseline with the SAME bias present: "
              f"{base * 100:.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="play N self-play games, write an .npz")
    r.add_argument("--load", required=True, help="checkpoint path or glob")
    r.add_argument("--games", type=int, default=100)
    r.add_argument("--envs", type=int, default=32, help="parallel envs")
    r.add_argument("--out", default="../runs/diagnosis.npz")

    a = sub.add_parser("analyze", help="analyse a recorded .npz")
    a.add_argument("--data", default="../runs/diagnosis.npz")
    a.add_argument("--contact-m", type=float, default=3.0,
                   help="inter-disc distance counted as an engagement")
    a.add_argument("--look", type=int, default=30,
                   help="decisions before a death to search for contact")
    a.add_argument("--swing-look", type=int, default=6,
                   help="window for the value-swing scan (archive.lookback)")

    n = sub.add_parser("arena", help="greedy vs sampled, three ways")
    n.add_argument("--load", required=True, help="checkpoint path or glob")
    n.add_argument("--games", type=int, default=200, help="games PER condition")
    n.add_argument("--envs", type=int, default=32)

    args = ap.parse_args()
    return {"record": record, "analyze": analyze, "arena": arena}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
