"""bonk2 interactive pygame playground — play or spectate the league.

You drive player 1 (cyan): arrows move/jump, X (or Left Shift) = heavy.
Player 2 (red) is idle by default, or driven from a training checkpoint:

  python -m bonk2.play --checkpoint CKPT.json                   # you vs main
  python -m bonk2.play --checkpoint CKPT.json --opponent exp:3  # you vs exploiter #3
  python -m bonk2.play --checkpoint CKPT.json --auto --opponent snap:0
                                          # spectate: main vs oldest snapshot

--opponent SPEC forms: main | snap | snap:N | exp | exp:N  (N indexes the pool;
omit N or use -1 for the newest). --auto hands player 1 to the main agent.

Both seats run with the training decision cadence (every ACTION_REPEAT ticks)
and the training self/view lags, so what you feel is what it trained in.
R = reset round, Esc = quit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pygame
import torch

from bonk import config as PC          # physics constants (kill line, heavy mass)
from bonk.networks import MLP
from bonk3.simadapter import make_sim

from .policy import load_policy, resolve_agent

from . import config as C
from .env import LagEnv

REPO = Path(__file__).resolve().parents[2]
WORLD_HEIGHT = 52.0  # meters shown vertically (matches the browser renderer)
BG = (5, 8, 10)
COL_P1 = (0, 229, 255)
COL_P2 = (255, 82, 82)
COL_KILL = (255, 40, 40)


def keys_to_index(k):
    lr = 1 if k["left"] else 2 if k["right"] else 0
    ud = 1 if k["up"] else 2 if k["down"] else 0
    return lr * 6 + ud * 2 + (1 if k["heavy"] else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"),
                    help="legacy-engine map file; ignored when config.ENGINE "
                         "== 'real' (that uses config.MAP_NAME)")
    ap.add_argument("--checkpoint", help="bonk2 training checkpoint JSON")
    ap.add_argument("--opponent", default="main",
                    help="who drives player 2: main | snap[:N] | exp[:N]")
    ap.add_argument("--auto", action="store_true",
                    help="drive player 1 (cyan) with the main agent too — "
                    "spectate bot-vs-bot")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    sim = make_sim(C.MAP_NAME if C.ENGINE == "real" else args.map, engine=C.ENGINE)
    env = LagEnv(sim)
    env.reset(randomize_spawns=False)

    p0_policy = None      # drives seat 0 (cyan) — you, unless --auto
    p1_label = "idle"
    p1_policy = None      # drives seat 1 (red)
    if args.checkpoint:
        with open(args.checkpoint) as f:
            ckpt = json.load(f)
        if ckpt.get("stateDim") not in (None, C.STATE_DIM):
            print(f"warning: checkpoint stateDim {ckpt.get('stateDim')} != "
                  f"{C.STATE_DIM} — is this a v1 model?")
        recs, label = resolve_agent(ckpt, args.opponent)
        p1_policy = load_policy(recs)
        p1_label = label
        print(f"player 2 (red): {label} [{p1_policy.kind}]")
        if args.auto:
            recs, _ = resolve_agent(ckpt, "main")
            p0_policy = load_policy(recs)
            print(f"player 1 (cyan): main agent (auto) [{p0_policy.kind}]")
    elif args.auto:
        ap.error("--auto needs --checkpoint")

    pygame.init()
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption("bonk2 sim — arrows + X/Shift heavy, R reset")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    # Screen centre — used for on-screen text in BOTH modes, so it must not
    # live inside a branch.
    cx, cy = screen.get_width() / 2, screen.get_height() / 2

    # ENGINE="real" draws the actual decoded bonk geometry (bonk3.render);
    # "legacy" keeps the old approximate platform list.
    real = C.ENGINE == "real"
    if real:
        from bonk3.render import Camera, MapRenderer
        rend = MapRenderer(sim.map_data)
        cam = Camera.fit(sim.map_data, screen.get_width(), screen.get_height())
        scale = cam.scale
        to_screen = cam.to_screen
    else:
        scale = screen.get_height() / WORLD_HEIGHT

        def to_screen(x, y):
            return int(cx + x * scale), int(cy + y * scale)

    score = [0, 0, 0]  # p1 wins, p2 wins, draws
    result_flash = ""
    flash_until = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    env.reset(randomize_spawns=False)
                    for pol in (p0_policy, p1_policy):
                        if pol is not None:
                            pol.reset_env(0)

        on_decision = env.episode_steps % C.ACTION_REPEAT == 0

        # Player 1 (cyan): the main agent under --auto, else you.
        if p0_policy is not None:
            if on_decision:
                env.set_decision(0, int(p0_policy.act(
                    env.decision_state(0), args.greedy)[0]))
        else:
            pressed = pygame.key.get_pressed()
            human = {
                "up": pressed[pygame.K_UP],
                "down": pressed[pygame.K_DOWN],
                "left": pressed[pygame.K_LEFT],
                "right": pressed[pygame.K_RIGHT],
                "heavy": pressed[pygame.K_x] or pressed[pygame.K_LSHIFT],
            }
            # Human decides every tick; the env applies it SELF_LAG ticks later.
            env.set_decision(0, keys_to_index(human))

        # Player 2 (red) decides on the training cadence.
        if p1_policy is not None and on_decision:
            env.set_decision(1, int(p1_policy.act(
                env.decision_state(1), args.greedy)[0]))

        res = env.tick()
        if res["done"]:
            if res["timeout"] or all(res["dead"]):
                score[2] += 1
                result_flash = "DRAW"
            elif res["dead"][1]:
                score[0] += 1
                result_flash = "CYAN WINS" if p0_policy else "YOU WIN"
            else:
                score[1] += 1
                result_flash = "RED WINS" if p1_policy else "YOU DIED"
            flash_until = pygame.time.get_ticks() + 1200
            env.reset(randomize_spawns=False)
            for pol in (p0_policy, p1_policy):
                if pol is not None:
                    pol.reset_env(0)     # new episode, no carried memory

        # ── draw ───────────────────────────────────────────────────────────
        screen.fill(BG)
        if real:
            rend.draw(screen, cam)
        for plat in ([] if real else sim.platforms):
            color = plat["color"] if plat["color"] is not None else 0x3A5A40
            rgb = ((color >> 16) & 255, (color >> 8) & 255, color & 255)
            if plat["kind"] == "box":
                x, y = to_screen(plat["x"], plat["y"])
                pygame.draw.rect(screen, rgb,
                                 (x, y, max(1, plat["w"] * scale),
                                  max(1, plat["h"] * scale)))
            elif plat["kind"] == "circle":
                pygame.draw.circle(screen, rgb, to_screen(plat["x"], plat["y"]),
                                   max(1, int(plat["radius"] * scale)))
            else:
                pts = [to_screen(plat["x"] + p[0], plat["y"] + p[1])
                       for p in plat["points"]]
                if len(pts) >= 3:
                    pygame.draw.polygon(screen, rgb, pts)

        if not real:   # the real engine reports deaths directly; no kill line
            ky = int(cy + (PC.KILL_LINE_Y / sim.ppm) * scale)
            pygame.draw.line(screen, COL_KILL, (0, ky), (screen.get_width(), ky), 1)
            pygame.draw.circle(screen, COL_KILL, (int(cx), int(cy)),
                               int((PC.KILL_RADIUS / sim.ppm) * scale), 1)

        for i, col in ((0, COL_P1), (1, COL_P2)):
            p = sim.players[i]
            # The renderer works in the engine's frame; player.pos is
            # centre-origin (see EnginePlayer.pos), so add the offset back.
            px, py = p.pos
            if real:
                px += sim.origin[0]
                py += sim.origin[1]
            x, y = to_screen(px, py)
            r = max(2, int(p.radius * scale))
            pygame.draw.circle(screen, col, (x, y), r)
            heavy_frac = ((p.heavy_power / 1000.0 if p.heavy_active else 0.0) if real
                          else max(0.0, min(1.0, (p.body.mass - 1) / PC.HEAVY_EXTRA_MASS)))
            if heavy_frac > 0.01:
                pygame.draw.circle(screen, (255, 255, 255), (x, y), r,
                                   max(1, int(r * 0.18 * heavy_frac + 0.5)))

        p1 = sim.players[0]
        cyan_name = "main" if p0_policy else "you"
        red_name = p1_label if p1_policy else "idle"
        hud = [
            f"score  cyan({cyan_name}) {score[0]} - {score[1]} red({red_name})"
            f"   draws {score[2]}",
            f"pos ({p1.pos[0]:6.2f},{p1.pos[1]:6.2f})  "
            f"vel ({p1.vel[0]:6.2f},{p1.vel[1]:6.2f})",
            f"heavy {p1.heavy_power:4.0f}"
            + ("" if real else f"  mass {p1.body.mass:4.2f}") + "  "
            f"tick {env.episode_steps}/{C.MAX_EPISODE_STEPS}",
            f"selfLag {env.self_lag}  viewLag {env.view_lag}  "
            f"cyan={cyan_name}  red={red_name}",
        ]
        for j, line in enumerate(hud):
            screen.blit(font.render(line, True, (200, 240, 230)), (8, 8 + j * 17))
        if pygame.time.get_ticks() < flash_until:
            big = pygame.font.SysFont("monospace", 48, bold=True)
            text = big.render(result_flash, True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(cx, cy - 100)))

        pygame.display.flip()
        clock.tick(C.TPS)

    pygame.quit()


if __name__ == "__main__":
    main()
