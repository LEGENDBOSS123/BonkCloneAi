"""Interactive pygame playground for the Python sim — feel-test the physics.

You drive player 1 (cyan): arrows move/jump, X (or Left Shift) = heavy.
Player 2 (red) is idle by default, or driven by a trained rl2 checkpoint:

  python -m bonk.play                          # opponent sits still
  python -m bonk.play --checkpoint SAVE.json   # opponent = AVG policy
  python -m bonk.play --checkpoint SAVE.json --net policy --greedy
                                               # opponent = BR, argmax actions

The opponent runs with the training-time decision cadence (every ACTION_REPEAT
ticks) and both seats get the training-time INPUT_LAG, so what you feel is what
the agent trained in. R = reset round, Esc = quit.
"""

import argparse
import json
from pathlib import Path

import pygame

from . import config as C
from .env2 import LagEnv
from .sim import BonkSim, index_to_keys
from .policy import MLPPolicy

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
    ap.add_argument("--map", default=str(REPO / "src/bonkmap/map1.json"))
    ap.add_argument("--checkpoint", help="rl2 save JSON; opponent uses it")
    ap.add_argument("--net", default="avg", choices=["avg", "policy", "actor"],
                    help="avg/policy = NFSP saves; actor = PPO saves")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    with open(args.map) as f:
        sim = BonkSim(json.load(f))
    env = LagEnv(sim)
    env.reset(randomize_spawns=False)

    policy = None
    if args.checkpoint:
        policy = MLPPolicy.from_checkpoint(args.checkpoint, args.net)
        print(f"opponent: {args.net} net from {args.checkpoint}")

    pygame.init()
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption("bonk python sim — arrows + X/Shift heavy, R reset")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    scale = screen.get_height() / WORLD_HEIGHT
    cx, cy = screen.get_width() / 2, screen.get_height() / 2

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

        pressed = pygame.key.get_pressed()
        human = {
            "up": pressed[pygame.K_UP],
            "down": pressed[pygame.K_DOWN],
            "left": pressed[pygame.K_LEFT],
            "right": pressed[pygame.K_RIGHT],
            "heavy": pressed[pygame.K_x] or pressed[pygame.K_LSHIFT],
        }
        # Human decides every tick; the env applies it INPUT_LAG ticks later.
        env.set_decision(0, keys_to_index(human))

        # Opponent decides on the training cadence.
        if policy and env.episode_steps % C.ACTION_REPEAT == 0:
            obs = env.decision_state(1)
            env.set_decision(1, policy.act(obs, greedy=args.greedy))

        res = env.tick()
        if res["done"]:
            if res["timeout"] or all(res["dead"]):
                score[2] += 1
                result_flash = "DRAW"
            elif res["dead"][1]:
                score[0] += 1
                result_flash = "YOU WIN"
            else:
                score[1] += 1
                result_flash = "AI WINS" if policy else "YOU DIED"
            flash_until = pygame.time.get_ticks() + 1200
            env.reset(randomize_spawns=False)

        # --- draw ---------------------------------------------------------
        screen.fill(BG)
        for plat in sim.platforms:
            color = plat["color"] if plat["color"] is not None else 0x3A5A40
            rgb = ((color >> 16) & 255, (color >> 8) & 255, color & 255)
            if plat["kind"] == "box":
                x, y = to_screen(plat["x"], plat["y"])
                pygame.draw.rect(screen, rgb,
                                 (x, y, max(1, plat["w"] * scale), max(1, plat["h"] * scale)))
            elif plat["kind"] == "circle":
                pygame.draw.circle(screen, rgb, to_screen(plat["x"], plat["y"]),
                                   max(1, int(plat["radius"] * scale)))
            else:
                pts = [to_screen(plat["x"] + p[0], plat["y"] + p[1]) for p in plat["points"]]
                if len(pts) >= 3:
                    pygame.draw.polygon(screen, rgb, pts)

        # kill line + circle
        ky = int(cy + (C.KILL_LINE_Y / sim.ppm) * scale)
        pygame.draw.line(screen, COL_KILL, (0, ky), (screen.get_width(), ky), 1)
        pygame.draw.circle(screen, COL_KILL, (int(cx), int(cy)),
                           int((C.KILL_RADIUS / sim.ppm) * scale), 1)

        for i, col in ((0, COL_P1), (1, COL_P2)):
            p = sim.players[i]
            x, y = to_screen(*p.pos)
            r = max(2, int(p.radius * scale))
            pygame.draw.circle(screen, col, (x, y), r)
            heavy_frac = max(0.0, min(1.0, (p.body.mass - 1) / C.HEAVY_EXTRA_MASS))
            if heavy_frac > 0.01:
                pygame.draw.circle(screen, (255, 255, 255), (x, y), r,
                                   max(1, int(r * 0.18 * heavy_frac + 0.5)))

        p1 = sim.players[0]
        hud = [
            f"score  you {score[0]} - {score[1]} ai   draws {score[2]}",
            f"pos ({p1.pos[0]:6.2f},{p1.pos[1]:6.2f})  vel ({p1.vel[0]:6.2f},{p1.vel[1]:6.2f})",
            f"heavy {p1.heavy_power:4.0f}  mass {p1.body.mass:4.2f}  "
            f"tick {env.episode_steps}/{C.MAX_EPISODE_STEPS}",
            f"lag {C.INPUT_LAG} ticks  opponent {'policy' if policy else 'idle'}",
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
