# Bonk RL — Project Chronicle

A chronological, detailed record of every approach tried to train a self-play RL
agent for a Bonk.io-clone 1v1 physics brawler, with results. Read top-to-bottom
for the story; each entry is **what → why → result**.

## The two hard constraints (held throughout)

1. **No hardcoded state-dependent rule rewards.** Outcome reward only
   (`+1`/`−1`/`−0.5` at round end), plus *state-independent* time cost, *learned*
   intrinsic reward, or *return-equivalent* transforms (PBRS/RUDDER). Test: could
   someone who never played bonk write this reward term?
2. **Minimal, map-agnostic observation.** ~30–33 normalized floats, no
   hand-coded map features. The agent must infer geometry from its own position.

---

## Era 0 — Foundations (pre-curriculum)

### 0.1 In-browser TF.js training (`src/rl`, `src/rl2`)
- **What:** train directly in the browser tab with TensorFlow.js against the live
  JS game.
- **Result:** worked as a proof of concept but slow and awkward; superseded by a
  Python pipeline. Kept only as the original game + deploy target.

### 0.2 Box2D physics port (`python/bonk/sim.py`)
- **What:** reimplement the JS game physics in Python (Box2D 2.3.10, Python 3.12).
- **Result:** **verified to match the JS sim to ~1e-5 m** via replay tests. This
  parity is the reason the whole train-in-Python / deploy-to-JS pipeline works.

### 0.3 `LagEnv` with input-lag simulation (`env2.py`)
- **What:** a decision-level env where a decision at tick `t` applies at `t+lag`,
  with `lag` randomized per episode (the sim-to-real bridge for real netcode).
- **Result:** load-bearing; kept in every later version.

### 0.4 PPO + AlphaStar-lite league (`train_ppo_mp.py`)
- **What:** PPO over 18 joint actions, `[256,256]` LayerNorm MLP; multiprocess
  collection; league with PFSP snapshot/exploiter pools.
- **Known open problems at this stage:**
  - **Cycling** (rock-paper-scissors, not monotone improvement).
  - **Juke vulnerability** — rushes and falls off when the opponent dodges.
  - Action-level exploration is weak for the *strategic* shifts the game needs.

---

## Era 1 — `ppo6`: the reward/curriculum investigation

The core problem: the agent **stalls** (passive) or **suicides** (drives off the
edge) instead of fighting well. Everything below is the hunt for why.

### 1.1 Draw-reward sweep
- **What:** try `DRAW = −1`, `−2`, `0` to break stalling.
- **Result:**
  - `−2` → **suicide** (dying became better than drawing) — *predicted and
    confirmed*.
  - `0` → **mutual stalling** in a symmetric game (p>0.5 commit threshold).
  - `−1` (draw == loss) is the **unique** point that removes the stalling refuge
    *without* a suicide incentive. Adopted as the baseline.

### 1.2 The 3-phase curriculum (the central plan)
- **What:** (1) actor learns to kill a *stationary* bot with a dense
  distance reward + **scalar** critic; (2) freeze the actor, a fresh
  **categorical** outcome critic learns in self-play; (3) unfreeze both — real
  league training begins.
- **Why:** death-from-scratch can't grow an "approach" drive because exploration
  dies before it reaches the opponent; bootstrap the drive on an easy target.
- **Result:** became the backbone for the rest of the project (see 1.9, Era 2).

### 1.3 Dense distance reward — five forms tried
The single most-iterated question. All are "distance to opponent," but the *form*
matters enormously:
- **Closeness occupancy `Φ(s)` as advantage add-on** → **no-op**: a far agent
  never samples close states, so no gradient toward engaging (measured ~0%
  contact).
- **Distance-decrease / approach-difference `Φ(s')−Φ(s)`** → rewards *motion*
  toward the opponent; standing still pays 0. This *worked* — the agent moved and
  approached.
- **Bounded 1/x potential `Φ = 1/(1+d/D)`** → same but bounded so it can't blow
  up at contact; can't be farmed by hugging (telescopes to net approach).
- **PBRS with terminal clawback** (`γΦ(s′)−Φ(s)`, `Φ(terminal)=0`) →
  **policy-invariant** (verified: discounted shaping sum = `−Φ(s₀)` for any
  path). Safe at any coefficient. This is the constraint-legal form.
- **Raw occupancy reward `Φ(s)` per step** → **stand-still pathology**: on
  fragile terrain, sitting in a decent spot collects safe reward while *moving*
  risks a fatal fall, so the agent **froze** (confirmed: 50% winrate = wins
  nothing, and it visibly stood still). Occupancy rewards the *status quo*, not
  improvement.
- **Verdict:** reward the *change* in distance, not the level.

### 1.4 The `death` map diagnosis — it's mostly a pit
- **What:** probed survivability of 30k random spawn positions.
- **Result:** **only ~4.5% of the map is survivable.** Two tiny top spawn
  platforms (y≈39) separated by a **lethal pit**, a middle platform with a
  **hole**, and a wide bottom floor. So "reduce euclidean distance to the
  opponent" literally points *into the pit* when the opponent is across it — the
  suicide gradient was the reward, not a policy bug.

### 1.5 Survivable-spawn pool (`gen_spawns.py`)
- **What:** drop discs at 1000s of random positions, idle 10 s, keep the ones
  that **survive** (detecting death *events*, not end-state — off-platform discs
  die-then-respawn and fool a naive check). Record 1448 stable ground points;
  sample both discs from them.
- **Result:** no more death-trap spawns; "self" now spans the whole map, which
  also **fixed the seat-1 generalization problem** (see 1.6) for free.

### 1.6 The seat-1 bug (why "it never reaches me")
- **What:** the agent trained only as seat 0; `play.py` put the AI on seat 1.
- **Result:** verified headless — seat 0 wins 4/8 vs a stationary target, seat 1
  wins **0/8**. Nothing wrong with the model; it was being tested on its
  untrained seat. Fixed by the spawn pool (self spans the map) + a `--swap` flag.

### 1.7 tanh vs LayerNorm
- **What:** suspected the LayerNorm+ReLU trunk hurt learning of absolute
  position (LayerNorm removes per-sample magnitude).
- **Result:** tested — the LayerNorm net **fit a spatial danger-map to 98%**, so
  capacity was fine. Switched to a plain **tanh MLP with biases** anyway
  (canonical PPO, and it made steps/s ~1.6× faster by dropping LayerNorm). It was
  *not* the bug.

### 1.8 Two silent correctness bugs found
- **γ mismatch:** `GAMMA=0.997` (right for the scalar-critic phase-1 with dense
  per-step reward) is **wrong** for the categorical outcome critic, whose target
  propagates *undiscounted* — so its value is a martingale and GAE is unbiased
  only at `γ=1`. `0.997` injected a `−0.003·V` per-step advantage bias. Fixed to
  `1.0`.
- **Critic outcome-labeling:** the categorical critic labeled the terminal class
  by **nearest-atom-to-reward**, but with `DRAW == LOSS == −1` that **merged draw
  and loss** — the critic never learned "loss" and read every losing position as
  a draw (visible in the play.py eval bar). Fixed to label from the **actual**
  win/draw/loss result. (Value was always correct = `2·P(win)−1`; the fix makes
  the displayed distribution honest.)

### 1.9 Entropy-schedule lessons
- **Warm-start wipe:** on phase 3, resetting the entropy coefficient to the usual
  high `0.1` **re-randomized** the warm-started actor (`H` jumped 0.77 → 2.78 in a
  few updates), discarding the whole phases-1/2 investment. Fix: **flat low
  entropy (0.003–0.01)** when refining a warm prior; a high start is only for
  from-scratch runs.
- **Loss-vs-draw risk aversion:** a "major" `LOSS=−3` (> the `−1` draw) made a
  *guaranteed draw safer than any risky drop*, so the agent **refused** the easy
  top→bottom kill. Reverted to `LOSS == DRAW == −1`. Lesson: you can't make losing
  "much worse than drawing" without inducing passivity, because you're not allowed
  to tell the agent which edge-departures are safe.

### 1.10 Operational: the orphaned-worker memory leak
- **What:** stopping runs with `pkill -f ppo6.train` killed only the main
  process; the multiprocessing workers (`python -c from multiprocessing.spawn…`)
  orphaned and accumulated (~350 of them, ~2 GB) → OOM'd the editor.
- **Fix:** always sweep `multiprocessing.spawn` workers with `ppid==1` after
  stopping a run.

### 1.11 ppo6 3-phase result
- **Phase 1:** ~75–92% kill-rate vs idle (dense reward built the approach prior).
- **Phase 2:** categorical critic converged on the frozen policy.
- **Phase 3:** ELO **1000 → 1313**; and — the honest metric — **`eval_h2h` of the
  phase-3 final vs its start = 83.5%** (both seats symmetric). Real improvement,
  not the self-pool ELO treadmill. The 3-phase curriculum **worked**.

---

## Era 2 — `ppo7`: dynamic action repetition (FiGAR)

### 2.1 The idea: `(action, duration)`
- **What:** the actor emits an action **and** a hold length from
  `{1,2,4,8,16,32,64}` cycles; the action is forced to repeat for that long. A
  second categorical head; joint log-prob/entropy add.
- **Why:** in an adversarial game, coherent multi-step commitments ("dash right
  for 16") are exactly the exploratory moves per-step noise never produces — the
  hypothesized fix for the pit-crossing.
- **Design choices:** categorical (not a weighted-sum/continuous duration, which
  has no clean PPO log-prob); the user's **decrement idea** — expose the remaining
  hold in the obs and treat every cycle as a transition — which turned a hard
  async semi-MDP into a plain per-step MDP that reuses the fast lockstep buffer.
  With `γ=1`, semi-MDP GAE collapses to ordinary GAE; the actor trains only on
  free-decision cycles, the critic on all cycles.
- **Verification:** hold accounting unit-tested (duration-16 ⇒ 1 free + 15 held;
  frees at cycles 0/16/32); two-head update sharpens; league (snapshots +
  exploiters), `play.py`, `eval_h2h` all made hold-aware.

### 2.2 Gang Grounds 2 (from scratch → league)
- **Result:** trained cleanly to **ELO 1884** (30 snapshots, 13 exploiters). But
  the duration probe showed it stayed **reactive**: 63% duration-1, 31%
  duration-2, **0% ≥8**. On a normal map with no pit, long holds aren't needed,
  and long commitments are exploitable in self-play — so short is optimal there.

### 2.3 `death` — the 3-phase curriculum with holds
- **Phase 1:** **92% kill-rate vs idle** (mean time-to-kill ~8 s, 1% self-death),
  clearly better than ppo6 on the same map.
- **Phase 2/3:** critic warmed; phase 3 ran ~41 h to **ELO 1742**, 12.7B steps.
- **Full stats (phase-3 final):** vs idle 90% win / 1% self-death / kill in ~7 s;
  self-play aggressive (~85% of games end in a knock-off, 15% draws); uses heavy
  ~25–30%; moves ~93–95% of the time.

### 2.4 The duration finding — and the correction
- **First read (too harsh):** the *final* policy uses only 1–2 cycle holds, **0%
  long**, in every setting → "temporal abstraction unused."
- **Correction (from the entropy trajectory):** early in phase 1, entropy sat at
  **3.3–4.8** (near the two-head max 4.84), meaning the duration head was
  **near-uniform**, so ~**57%** of early decisions were long holds (≥8). Those are
  the coherent commitments — and win-rate climbed 12→56% *during* that
  high-entropy period, then **leapt 56→90% exactly as entropy collapsed** and the
  policy annealed durations down to short/reactive.
- **Conclusion:** the holds did their job **during exploration**, then got tuned
  out for reactive precision — which is exactly how FiGAR is meant to work. The
  clean causal proof would be an A/B vs a duration-forced-to-1 baseline; the fast
  phase-1 convergence + 92% (vs ppo6's ~64%) is circumstantial support.

---

## Era 3 — Deployment

### 3.1 `play7.mjs` (browser deploy for ppo7)
- **What:** paste-into-tab script; rebuilds the two-head tanh actor in plain JS
  (no TF.js/CDN), reads live bonk.io state, drives the player with FiGAR holds.
- **Key correctness points:** the obs is **33 dims, no time feature** (the bug
  that broke the older play3.mjs, which assumed 34 with a time slot); the net is
  tanh **with biases** and **no LayerNorm**; the actor only queries on a *free*
  cycle (so the appended hold feature is always 0 at deploy).
- **Verified headless** (`test_play7.mjs`): JS forward matches PyTorch to
  **7.6e-6**; two-head split correct; hold pattern exact (`FHHHF` for a 4-cycle
  hold).

---

## Era 3.5 — Any map, on autopilot

### 3.5.1 Adding a map
- **What:** decode a bonk.io map with `bonk3/tools/decode_maps.mjs` (by name,
  DB index, or your own share code) → `bonk3/maps/<slug>.json`; set
  `MAP_NAME`; recompute survivable spawns with `gen_spawns.py` (per-map sampling
  box, found from a quick survivability scan).
- **Applied to `somewhat large burger`** (MaeIstrom): a tiered map (top y≈30 /
  middle y≈21 / bottom y≈12, ~21% survivable, less pit-heavy than `death`). Pool
  of 2279 survivable points.

### 3.5.2 Unattended 3-phase orchestrator (`run_burger_pipeline.sh`)
- **What:** a shell orchestrator runs P1 → P2 → P3 with **no user input**,
  chaining checkpoints automatically and `sed`-flipping `ENTROPY_COEF_START`
  0.1 → 0.01 before P3 (the warm-prior anti-wipe).
- **Safety:** before committing each multi-hour phase it VERIFIES the trainer's
  printed `phase flags` line against an expected pattern and asserts the
  config-file values, **aborting rather than wasting hours on a wrong config**.
- **Result:** ran end-to-end — phase 1 hit high kill-vs-idle, phase 3 climbed to
  ELO ~1579+ on burger.

## Era 4 — Turtle-breaker: staller exploiters (the decisive one)

The single most impactful change measured. Directly attacks the oldest weakness
in the project: **the agent can't close out a passive/turtling opponent** (the
juke/turtle problem from Era 0).

### 4.1 The idea (user's)
Sometimes spawn exploiters whose goal is to **time the main out** (turtle to a
draw), and when the main later plays one, score the main's **timeout as 0
(neutral)** instead of −1 — so the gradient points at the *kill*, not at avoiding
the staller.

### 4.2 The subtle bug caught before it shipped
Naively setting the main's timeout return to 0 **backfires**: the categorical
critic values a draw at its −1 atom and **cannot tell a staller game from a
normal one** (opponent type isn't in the map-agnostic obs). So a 0 return against
a −1-expecting value reads as a **+1 advantage** — it would *reward* the main for
drawing the turtle, the exact opposite of the intent.

### 4.3 The fix — per-opponent value re-weighting
The critic still predicts the win/draw/loss **distribution** (opponent-agnostic);
vs a staller the **value** is re-weighted with a draw-atom of 0:
`V = P(win)·1 + P(draw)·0 + P(loss)·(−1) = V_normal + P(draw)`, matching the 0
return. Verified: a staller-draw gives advantage **−0.02 (neutral) with the fix**
vs **+0.36 (rewards drawing) without it**.

### 4.4 Implementation
- 25% of exploiters are stallers (`EXPLOITER_STALLER_PROB=0.25`); a staller trains
  with draw-as-win atoms `(1,1,−1)` and a `+1` reward per draw, so it learns to
  turtle. Graduated stallers are **tagged** in the pool (persists through
  save/load). The main's timeout vs a tagged staller → 0, with the value
  re-weight above (`_value_staller`, `MAIN_VS_STALLER_ATOMS`).
- Continued from the burger phase-3 checkpoint (ELO 1579) into a new folder.

### 4.5 Result — the isolating test
Both mains played an **actual graduated staller turtle** (2000 games each):

| main | kills the turtle | **draws (fails)** | losses |
|---|---|---|---|
| **Old** (pre-staller) | **6.5%** | **81%** | 12.5% |
| **Staller-trained** | **57.9%** | 34% | 8% |

The old main **literally cannot finish a turtle** (81% draws); after staller
training the main **kills the same turtle 58%** of the time. Head-to-head overall,
the staller main beats the old fork **68.7%**. This is a *specific* gain at the
targeted skill, not just "more training" — the clearest single win in the project.

---

## Standing conclusions

1. **The pipeline works** and produces strong, aggressive agents; the 3-phase
   curriculum (approach prior → warm categorical critic → unfreeze) is the
   reliable recipe, validated by `eval_h2h`-vs-past (not self-pool ELO).
2. **Reward form >> reward magnitude.** Reward the *change* in distance
   (approach), never the *level* (occupancy → stand-still); keep it
   potential-based to stay policy-invariant; keep `DRAW == LOSS` to avoid both
   stalling and suicide.
3. **`γ=1` for the outcome critic; flat-low entropy when refining a warm prior.**
4. **FiGAR holds help exploration, then anneal away** — not "unused," but not the
   long-horizon strategy layer either.
5. **Targeted sparring partners beat generic self-play for specific holes.**
   Staller exploiters + a neutral-timeout advantage fixed the turtle weakness
   (6.5% → 58% kill vs a turtle) that years of normal league play never closed.
   The catch: any per-opponent reward change must stay **critic-consistent** —
   re-weight the value's draw atom to match the return, or you accidentally
   reward the very behavior you're punishing.
6. **The remaining untested lever with real upside is planning/search** — MCTS in
   the *real* simulator (no learned-model error), distilled back into the fast
   reactive net. That adds genuine lookahead rather than re-parameterizing a
   reactive policy, and is the natural next direction.
