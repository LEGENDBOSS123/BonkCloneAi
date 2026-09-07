"""Pygame playground: watch an agent, or play against it.

    python -m ppo9.play --checkpoint ../runs/gg9-p3/ppo9-*.json          # you vs it
    python -m ppo9.play --checkpoint ... --auto                          # it vs itself
    python -m ppo9.play --checkpoint ... --opponent snap:-1              # vs a snapshot

Controls: arrows move, X or Shift is heavy, R resets, Esc quits.

The eval bar reads the OPPONENT seat's win probability straight off the
categorical critic — i.e. the AI's own belief about how it is doing, which is
usually more informative than watching the discs.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from bonk3.simadapter import make_sim

from bonkenv import EnvConfig, LagEnv, Outcome, build_layout
from .config import FigarConfig
from .mingru import MinGRUNet
from .policy import load_checkpoint, load_policy, resolve_agent

COL_BG = (24, 26, 32)
COL_P1 = (80, 220, 255)
COL_P2 = (255, 82, 82)
COL_TXT = (220, 220, 230)


def keys_to_index(k: dict[str, bool]) -> int:
    """Human key state -> joint action index (inverse of `index_to_keys`)."""
    lr = 1 if k["left"] else 2 if k["right"] else 0
    ud = 1 if k["up"] else 2 if k["down"] else 0
    return lr * 6 + ud * 2 + (1 if k["heavy"] else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--agent", default="main", help="main | snap:N | exp:N")
    ap.add_argument("--opponent", default=None,
                    help="a second agent spec; implies --auto")
    ap.add_argument("--auto", action="store_true", help="agent vs agent, no human")
    ap.add_argument("--swap", action="store_true", help="human plays seat 1")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import pygame
    from bonk3.render import Camera, MapRenderer

    obj, path = load_checkpoint(args.checkpoint)
    figar = FigarConfig()
    recs_a, label_a = resolve_agent(obj, args.agent)
    pol_a = load_policy(recs_a, figar)

    auto = args.auto or args.opponent is not None
    pol_b = None
    label_b = "human"
    if auto:
        recs_b, label_b = resolve_agent(obj, args.opponent or args.agent)
        pol_b = load_policy(recs_b, figar)

    # Seat assignment: the human (if any) takes seat 0 unless --swap.
    human_seat = None if auto else (1 if args.swap else 0)
    if human_seat == 0:
        p0, p1 = None, pol_a
    elif human_seat == 1:
        p0, p1 = pol_a, None
    else:
        p0, p1 = pol_a, pol_b
    for p in (p0, p1):
        if p is not None:
            p.reset(1)

    # The critic, for the eval bar. Optional — a checkpoint always has one.
    critic = None
    try:
        critic = MinGRUNet.from_records(obj["agent"]["critic"])
        critic.eval()
        atoms = torch.tensor(obj.get("config", {}).get("critic", {}).get(
            "atoms", [1.0, -1.0, -1.0]), dtype=torch.float32)
        crit_h = critic.zero_h(1)
    except Exception:
        critic = None

    cfg = EnvConfig().resolve()
    layout = build_layout(cfg.obs)
    sim = make_sim(cfg.engine.map_name, engine=cfg.engine.engine)
    env = LagEnv(sim, cfg, layout)
    env.reset()

    # The eval bar shows the NON-human seat's win probability.
    eval_seat = 1 if human_seat in (None, 0) else 0

    pygame.init()
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption("ppo9 — arrows + X/Shift heavy, R reset, Esc quit")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    rend = MapRenderer(sim.map_data)
    cam = Camera.fit(sim.map_data, screen.get_width(), screen.get_height())

    score = [0, 0, 0]           # seat0 wins, seat1 wins, draws
    flash, flash_until = "", 0
    win_p = 0.5
    crit_probs = [0.0, 0.0, 0.0]     # raw critic distribution [win, draw, loss]
    crit_logits = [0.0, 0.0, 0.0]    # ...and the head output behind it
    crit_v = 0.0                     # V = probs . atoms, what GAE consumes
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    env.reset()
                    for p in (p0, p1):
                        if p is not None:
                            p.reset_env(0)
                    if critic is not None:
                        crit_h = critic.zero_h(1)

        on_decision = env.episode_steps % cfg.episode.action_repeat == 0
        if on_decision:
            for seat, pol in ((0, p0), (1, p1)):
                if pol is not None:
                    env.set_decision(seat, int(pol.act(
                        env.decision_state(seat), args.greedy)[0]))
            if critic is not None:
                # The critic sees the same input the trainer fed it: the
                # observation plus that seat's remaining-hold feature.
                pol_seat = p1 if eval_seat == 1 else p0
                hold = 0.0 if pol_seat is None or pol_seat.hold is None else float(
                    pol_seat.hold[0]) / float(max(figar.durations))
                x = torch.from_numpy(np.concatenate([
                    env.decision_state(eval_seat), [hold]]).astype(np.float32))[None]
                with torch.no_grad():
                    out, crit_h = critic.step(x, crit_h)
                    probs = torch.softmax(out, -1)[0]
                # Keep the RAW head output as well as the distribution: the
                # logits say how confident the critic is in a way the softmax
                # hides, and V is what GAE and the value-swing archive actually
                # consume, so the bar alone is not the whole story.
                crit_logits = [float(v) for v in out[0]]
                crit_probs = [float(v) for v in probs]
                crit_v = float(probs @ atoms)
                win_p = crit_probs[int(Outcome.WIN)]

        if human_seat is not None:
            pressed = pygame.key.get_pressed()
            env.set_decision(human_seat, keys_to_index({
                "up": pressed[pygame.K_UP], "down": pressed[pygame.K_DOWN],
                "left": pressed[pygame.K_LEFT], "right": pressed[pygame.K_RIGHT],
                "heavy": pressed[pygame.K_x] or pressed[pygame.K_LSHIFT]}))

        res = env.tick()
        if res.done:
            if res.outcome[0] == Outcome.WIN:
                score[0] += 1
                flash = "CYAN WINS" if human_seat != 0 else "YOU WIN"
            elif res.outcome[0] == Outcome.LOSS:
                score[1] += 1
                flash = "RED WINS" if human_seat != 1 else "YOU WIN"
            else:
                score[2] += 1
                flash = "DRAW"
            flash_until = pygame.time.get_ticks() + 1200
            env.reset()
            for p in (p0, p1):
                if p is not None:
                    p.reset_env(0)
            if critic is not None:
                crit_h = critic.zero_h(1)

        # ── draw ───────────────────────────────────────────────────────────
        screen.fill(COL_BG)
        rend.draw(screen, cam)
        for seat, col in ((0, COL_P1), (1, COL_P2)):
            pl = sim.players[seat]
            # `EnginePlayer.pos` is CENTRE-ORIGIN (the engine's own map-centre
            # offset removed — see simadapter.py, done so the observation
            # matches play3.mjs). The renderer (Camera/MapRenderer) works in
            # the engine's raw frame, so the offset has to be added back here
            # or every disc renders shifted by -sim.origin. ppo7/play.py hits
            # the same seam; this is that same fix.
            px, py = pl.pos[0] + sim.origin[0], pl.pos[1] + sim.origin[1]
            # `pl.radius` is the actual physics disc radius (1.0m, EnginePlayer
            # in simadapter.py) — NOT bonk3.render.draw_discs's `radius_m=1.25`
            # default, which is a generic drawing-helper default rather than
            # this engine's real value. Using it hardcoded drew every disc 25%
            # too large.
            r = max(3, int(pl.radius * cam.scale))
            pos = cam.to_screen(px, py)
            pygame.draw.circle(screen, col, pos, r)
            # White heavy-charge ring, same convention as ppo7/play.py:
            # `heavy_power` is the RESERVE (sits at ~full when idle), so it
            # cannot indicate "is heavy on" by itself — must gate on
            # `heavy_active` (the engine's actual ability_active flag).
            if pl.heavy_active:
                heavy_frac = pl.heavy_power / 1000.0
                if heavy_frac > 0.01:
                    pygame.draw.circle(screen, (255, 255, 255), pos, r,
                                       max(1, int(r * 0.18 * heavy_frac + 0.5)))

        seat_name = "cyan" if eval_seat == 0 else "red"
        hud = [
            f"A={label_a}  B={label_b}   {path.split('/')[-1]}",
            f"cyan {score[0]}  red {score[1]}  draw {score[2]}",
            f"tick {env.episode_steps}  self_lag {env.self_lag} view_lag {env.view_lag}",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, COL_TXT), (12, 12 + 18 * i))

        if critic is not None:
            # Win-probability bar, from the eval seat's own critic.
            bw, bh, bx, by = 240, 16, screen.get_width() - 260, 16
            pygame.draw.rect(screen, (60, 62, 70), (bx, by, bw, bh))
            pygame.draw.rect(screen, COL_P1 if eval_seat == 0 else COL_P2,
                             (bx, by, int(bw * win_p), bh))
            # RAW critic readout, not just the bar. The bar collapses a 3-class
            # distribution to one number; these are the numbers the trainer
            # itself consumes.
            #   p(win/draw/loss)  the categorical head, post-softmax
            #   logits            pre-softmax, so saturation is visible where
            #                     the probabilities have already clipped to
            #                     ~1.000/0.000 and stopped being informative
            #   V                 probs . atoms — what GAE and the value-swing
            #                     archive actually read
            pw, pd, pl_ = crit_probs
            lines = [
                f"{seat_name} critic  V {crit_v:+.4f}   atoms "
                f"({', '.join(f'{float(a):+g}' for a in atoms)})",
                f"  p  win {pw:.4f}  draw {pd:.4f}  loss {pl_:.4f}",
                f"  z  win {crit_logits[0]:+7.3f}  draw {crit_logits[1]:+7.3f}"
                f"  loss {crit_logits[2]:+7.3f}",
            ]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, COL_TXT),
                            (bx - 140, by + bh + 4 + 16 * i))

        if flash and pygame.time.get_ticks() < flash_until:
            big = pygame.font.SysFont("monospace", 44, bold=True)
            surf = big.render(flash, True, COL_TXT)
            screen.blit(surf, surf.get_rect(center=(screen.get_width() // 2,
                                                    screen.get_height() // 2)))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
