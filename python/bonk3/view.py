"""Play/inspect a REAL bonk.io map on the real engine.

    python -m bonk3.view --map gang-grounds-2-0
    python -m bonk3.view --map food-fight --shot /tmp/food.png   # headless PNG

Arrows move, X / Left-Shift = heavy, R resets, Tab cycles maps, Esc quits.
Player 2 idles — this is a physics/geometry viewer, not the RL playground
(that stays in bonk2). Its job is to prove the engine and the map parsing are
right, and to let you look at any of the 2000+ decoded maps.
"""

from __future__ import annotations

import argparse
import os
import sys

import pygame

from .engine import BonkEngine
from .maps import available, load_map, summary
from .render import BG, Camera, MapRenderer, draw_discs


def keys_to_input(pressed) -> dict:
    return {
        "up": pressed[pygame.K_UP],
        "down": pressed[pygame.K_DOWN],
        "left": pressed[pygame.K_LEFT],
        "right": pressed[pygame.K_RIGHT],
        "action": pressed[pygame.K_x] or pressed[pygame.K_LSHIFT],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None, help="map stem (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=750)
    ap.add_argument("--shot", help="render one frame to this PNG and exit (headless)")
    ap.add_argument("--frames", type=int, default=30,
                    help="frames to simulate before --shot")
    ap.add_argument("--zoom", type=float, default=None,
                    help="world height in metres (default 52, the game's view)")
    ap.add_argument("--geometry", action="store_true",
                    help="fit the whole map instead of a fixed view height")
    ap.add_argument("--follow", action="store_true", help="camera tracks players")
    args = ap.parse_args()

    maps = available()
    if args.list or not maps:
        print("decoded maps:", ", ".join(maps) if maps else
              "(none — run node python/bonk3/tools/decode_maps.mjs --name \"...\")")
        return
    name = args.map or maps[0]
    idx = maps.index(name) if name in maps else 0

    if args.shot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("bonk3 — real engine, real maps")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    def load(i):
        m = load_map(maps[i])
        eng = BonkEngine(m)
        st = eng.reset(seed=1234.0)
        cam = (Camera.fit_geometry(m, args.width, args.height) if args.geometry
               else Camera.fit(m, args.width, args.height, args.zoom))
        return m, eng, MapRenderer(m), cam, st

    map_data, eng, rend, cam, state = load(idx)
    print(summary(map_data))

    running, frame = True, 0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    state = eng.reset(seed=1234.0)
                elif ev.key == pygame.K_TAB:
                    eng.close()
                    idx = (idx + 1) % len(maps)
                    map_data, eng, rend, cam, state = load(idx)
                    print(summary(map_data))

        pressed = pygame.key.get_pressed()
        state = eng.step([keys_to_input(pressed), None])
        if state.deaths:
            state = eng.reset(seed=float(frame))

        if args.follow:
            cam.follow(state.discs)
        screen.fill(BG)
        rend.draw(screen, cam)
        rend.draw_spawns(screen, cam)
        draw_discs(screen, cam, state.discs)

        p0 = state.discs[0] if state.discs else None
        hud = [
            f"{map_data['metadata']['name']} by {map_data['metadata']['author']}"
            f"   [{idx + 1}/{len(maps)}]  Tab=next",
            f"shapes {len(map_data['physics']['shapes'])}  "
            f"bodies {len(map_data['physics']['bodies'])}  "
            f"ppm {map_data['physics']['pixelsPerMeter']}",
            (f"p0 ({p0.x:7.2f},{p0.y:7.2f}) v({p0.vx:6.2f},{p0.vy:6.2f}) "
             f"heavy {p0.energy:4.0f}" if p0 else ""),
            f"frame {state.round_frames}"
            + ("  FROZEN (round start)" if state.frozen else ""),
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (200, 240, 230)), (8, 8 + i * 17))

        pygame.display.flip()
        frame += 1

        if args.shot:
            if frame >= args.frames:
                pygame.image.save(screen, args.shot)
                print(f"wrote {args.shot}")
                running = False
        else:
            clock.tick(30)

    eng.close()
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
