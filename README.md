# BonkCloneAI

A reinforcement-learning agent that learns to play a [Bonk.io](https://bonk.io)-style
1v1 physics brawler **from scratch through self-play**, with two deliberate hard
constraints:

- **Sparse rewards only** — the agent gets `+1` win / `−1` loss / `−0.5` draw at the
  end of a round, and nothing else. No reward shaping.
- **Minimal, map-agnostic input** — a 30-number observation (both players'
  position/velocity/keys + a draw clock). No hand-coded, map-specific features.

The experiment: can that setup, plus league-style self-play, produce something that
beats real humans? A Python Box2D port of the game physics (verified bit-for-bit
against the JS sim) makes fast headless training possible; a paste-into-the-tab
script then runs the trained agent on the real game.

## Repo layout

| Path | What it is |
|------|-----------|
| `src/` | The JS Bonk clone — physics (`physics.mjs`), entities, map parser, Pixi rendering. `index.html` plays it. |
| `python/bonk/` | **The main RL project.** Box2D sim, env, PPO + league training, eval, deploy export. |
| `play2.mjs` | Paste-into-the-console script that runs a trained model on the live game. |
| `models/` | Slim, ready-to-play exported model(s) (~1 MB). |
| `src/rl/`, `src/rl2/`, `train*.html` | Earlier in-browser (TF.js) training experiments — superseded by the Python path, kept for reference. |

Full training checkpoints (100+ MB each, in `runs/`) are **git-ignored**; a slim
model is shipped in `models/` instead (see [Deploy](#play-against-the-real-game)).

## Quickstart

### Play / watch the browser clone
ES modules need a server (not `file://`):
```bash
npx serve .        # or: python3 -m http.server
# open the printed URL, then index.html
```

### Train (Python)
Needs **Python 3.12** (the `Box2D==2.3.10` wheel).
```bash
cd python
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m bonk.train_ppo_mp --workers 8        # CPU multiprocess self-play + league
```
Checkpoints land in `runs/ppo-mp/`. Ctrl-C saves and exits; `--load <ckpt>` resumes.

### Measure real progress (not ELO)
ELO is measured against a rolling buffer of the agent's own recent selves, so it
plateaus whether or not the agent is improving. To actually know, play two
checkpoints head-to-head:
```bash
cd python
python -m bonk.eval_h2h --a <newer_ckpt.json> --b <older_ckpt.json> --games 2000
```

### Play against the real game
1. Export a slim model (already done for the shipped one):
   ```bash
   cd python
   python -m bonk.export_model runs/ppo-mp/<checkpoint>.json --out ../models/bonk-ppo-latest.json
   ```
2. Open bonk.io, start a 1v1, open DevTools → Console, paste the contents of
   `play2.mjs`, and pick `models/bonk-ppo-latest.json` when prompted.
   Stop with `top.bonkai.playStop()`.

> This is a research/hobby project — please only run the agent in private rooms
> against yourself or friends, not to grief public games.

## How it works (short version)

- **Sim** (`bonk/sim.py`): a line-for-line Box2D port of the JS physics, verified to
  match to ~1e-5 m so a browser-trained-equivalent policy transfers.
- **Env** (`bonk/env2.py`): decision-level wrapper with simulated **input lag**
  (randomized per episode) so timing learned in sim survives the real game.
- **Algorithm** (`bonk/ppo.py`, `bonk/train_ppo_mp.py`): PPO over 18 joint actions
  with **AlphaStar-lite league training** — periodic *exploiters* best-respond to the
  frozen main to find its weaknesses, then the main trains against a
  winrate-prioritized (PFSP) mix of its past selves and those exploiters.

See [`CLAUDE.md`](./CLAUDE.md) for a deeper architecture map and the open problems.

## License
Personal project shared for tinkering. Bonk.io is a trademark of its owner; this is
an unaffiliated fan reimplementation for research.
