# ppo7 — feature catalog for a future rebuild

Every non-obvious design decision in the current PPO stack, as of 2026-09-02, with
what it is, why it was added, and what evidence (if any) exists for keeping it. Written
so a future rebuild can decide feature-by-feature instead of re-deriving all of this
from git history. Complements `ARCHITECTURE.md` (code map / module layout) and the
top-level `CLAUDE.md` (the two hard constraints: no state-dependent rule rewards, a
minimal map-agnostic observation — every reward-shaping feature below was checked
against that rule at the time it was added).

Status tags: **LIVE** (on by default, training with it right now) / **DISABLED**
(implemented, currently off) / **BUILT, UNUSED** (implemented, never wired into the
live config) / **INFRA** (not a modeling choice — perf/tooling only).

## Rebuild decisions (2026-09-02)

Asked feature-by-feature; this is the answer key for a future rebuild. ✅ = carry
over, ❌ = leave out.

| Feature | Decision |
|---|---|
| Categorical 3-atom outcome critic | ✅ keep |
| GAMMA=1 (undiscounted) | ✅ keep |
| Multi-gamma auxiliary critic heads | ✅ keep |
| Legacy position/time aux heads | ❌ drop |
| minGRU recurrent encoder | ✅ keep |
| FiGAR duration head | ✅ keep |
| Hold bonus | ❌ drop |
| Adaptive entropy controller | ❌ drop |
| AlphaStar-lite league (snapshots + PFSP) | ✅ keep |
| Exploiter phase (best-response training) | ✅ keep |
| Staller exploiters | ✅ keep |
| Tree search / lookahead at train time | ❌ drop |
| RUDDER return redistribution | ❌ drop |
| HCA (Hindsight Credit Assignment) | ❌ drop |
| Mirror (horizontal symmetry) augmentation | ✅ keep |
| Spawn curriculum + dense approach shaping | ✅ keep |

Not asked directly (follow from a kept feature they're wired to, or are pure infra —
call these out explicitly if the rebuild wants to reconsider them independently):
duration-entropy separate coefficient/floor (wired to FiGAR, kept), param-noise
exploiter exploration (wired to the exploiter phase, kept), fused backward / MPS
padding / `--force-exploiter` / `SingleProcessCollector` (infra, §5).

**Net effect on the reward/credit-assignment stack**: with RUDDER and HCA both
dropped, advantages go back to plain GAE off the categorical critic with no extra
return-redistribution or hindsight-credit term — the two credit-assignment
experiments this project tried are both out. The categorical critic + multi-gamma
aux heads are the only critic-side additions carried forward.

---

## 1. Core architecture

**minGRU recurrent encoder** — LIVE. **[rebuild: ✅ keep]** `mingru.py`. Linear recurrence
(`h_t=(1-z_t)h_{t-1}+z_t h̃_t`, input-only gates) instead of GRU/LSTM/transformer —
stable gradients, gates for a whole sequence batch in one matmul, only the elementwise
scan is sequential. Single-frame observation; temporal memory lives in the carried
hidden state (`h_a`/`h_c` per env), not in a stacked window. Replaced an earlier TCN
(`TemporalNet`, dilated causal conv over 16 raw frames) — TCN is kept in `networks.py`
as `RECURRENT=False` but is not what trains. Measured ~3x steps/s over TCN at the time
of the switch.

**FiGAR duration head** — LIVE. **[rebuild: ✅ keep]** `figar.py`, `config.DURATIONS=(1,2,4,8,16,32,64)`. The
actor emits a joint action AND a hold-duration; the action repeats for that many
decision-cycles. Gives temporally-coherent exploration a single per-cycle categorical
sample can't produce, and lets the agent pick its own reaction timescale.

**Categorical (3-atom) outcome critic** — LIVE. **[rebuild: ✅ keep]** `config.CRITIC_CATEGORICAL=True`,
`CRITIC_ATOMS=(WIN_REWARD, DRAW_REWARD, LOSS_REWARD)`. The critic classifies P(win/
draw/loss) instead of regressing a scalar return — exact only at `GAMMA=1` (terminal-
only reward), where the distributional Bellman backup is a plain backward propagation
of the realized outcome (`targets.outcome_targets`). Structurally bounded in
`[LOSS_REWARD, WIN_REWARD]` (softmax·atoms is a convex combination) — can't diverge,
unlike a scalar regressor's early-training value-loss blowup, which is what wrecked an
earlier shared-trunk architecture. **`GAMMA=1` is a hard requirement of this, not an
independent choice** — discounting would make V depend on time-to-end, which 3 class
probabilities can't represent.

## 2. Reward / advantage shaping

**RUDDER return redistribution** — LIVE **[rebuild: ❌ drop]** (`RUDDER_COEF=0.2`, `RUDDER_LOSS_COEF=0.5`).
Arjona-Medina et al. 2019. `rudder_head` rides the critic's own trunk (zero extra
forward pass), predicts g(s) = eventual outcome from a partial trajectory, trained by
regression against the realized outcome. Reward fed into GAE is redistributed as
`g[t]-g[t-1]` within an episode — added to the real reward, not replacing it. Applies
to the MAIN agent too (CLAUDE.md explicitly names RUDDER as an allowed
return-equivalent transform, unlike hand-written rewards). **Open caveat, never
resolved:** this project's critic is already RUDDER-like on its own — GAMMA=1 +
terminal-only reward means GAE's delta is already `V(s_{t+1})-V(s_t)`, and the
categorical critic's targets are already Monte-Carlo backward-propagated. The marginal
win over what's already there was never confirmed either way before it shipped.

**Hindsight Credit Assignment (HCA)** — DISABLED **[rebuild: ❌ drop]** (`config.HCA=False`), implementation
kept in place. Learns h(a|s,z) and credits an action by how much knowing the outcome
raises its probability. Tried on exploiters (both the staller-only and then all-
exploiter scope): **confirmed to reproduce a sharpening collapse on a combat
exploiter** — action entropy crashed to ~0.43 (of ln(18)=2.89), `decided%~9`,
`hold~13.6`, stuck at 37-40% winrate the whole run. Mechanism has a built-in positive
feedback loop (pi(a|s) sits in the ratio's denominator, so sharpening inflates the
ratio, which sharpens further). RUDDER was picked as the replacement. Staller-only
scope was never independently re-tested after being widened to all exploiters — the
one hypothesis that stalling-credit is *aligned* (not degenerate) for a staller
specifically remains untested, not disproven.

**Multi-gamma auxiliary critic heads** — currently OFF **[rebuild: ✅ keep]** (`AUX_GAMMA_COEF=0.0`, was
`0.1`), implementation kept live/wired. Fedus et al. 2019 pattern: extra scalar heads
on the critic trunk regressed at gamma=0.9/0.99, auxiliary-only (never touches the
advantage or policy gradient). **Actively contested as of this writing**: the first
exploiter run under `AUX_GAMMA_COEF=0.1` bottomed at 12% winrate ~9M steps in, deeper
than the prior (no-gamma-heads) exploiter's worst point of 25% across its entire run —
suggestive that a freshly-initialized aux head regressing into the shared trunk
destabilizes early training, though it also fully recovered (35-48%+ by 30M steps), so
this isn't fully confirmed as causal yet. `config.py`'s own notes on the two
*legacy* aux heads below already document a poor track record for this general idea
on this problem.

**Legacy aux heads: position + time-to-end** — LIVE but **[rebuild: ❌ drop]** coefficients are 0
(`CRITIC_TIME_COEF=0.0`, `AUX_NEXT_POS_COEF=0.0`), i.e. constructed, inert, never
enabled in practice. `time_head` predicts decisions-remaining (Lc0 MLH analogue);
`aux_head` predicts own future displacement at several horizons. Both ride the critic
trunk like RUDDER/gamma heads do. **Measured and NOT adopted**: time-to-end gave only
+0.030 representation lift (vs +0.219 for the outcome head itself); the position aux
head came back 27 points WORSE head-to-head. Kept as the standing evidence that
"just add an auxiliary head to the critic trunk" has a bad hit rate here, cited by both
RUDDER's and the multi-gamma heads' own risk write-ups before they were tried anyway.

**Hold bonus** — DISABLED **[rebuild: ❌ drop]** (`config.HOLD_BONUS=0.0`, was `0.01`). A state-independent
`+β·E[log2 hold]` term on the actor loss (not the reward), meant to teach the agent
*when* holding is free (the return gradient still pushes holds short wherever
re-deciding matters). Flagged as a likely — not proven in isolation — contributor to a
passive-hold drift on a combat exploiter that persisted even with HCA fully off
(hold 3.9→7.8, decided% 26%→14%, stuck ~38-40% winrate): it's the one remaining
state-independent force pushing toward longer holds, and unlike the entropy
coefficients it doesn't decay over a run, so its relative pull grows as opposition
weakens late in a schedule.

**Dense reward / approach shaping** — LIVE only in phase-1 warm start **[rebuild: ✅ keep]**
(`DENSE_REWARD`, `DENSE_COEF`, `targets.approach_shaping`, annealed via
`SHAPING_ANNEAL_STEPS`). Rewards reducing distance to the opponent (not being close —
that gave no gradient at ~0% contact rate). State-independent-in-the-allowed-sense:
it's a shaping potential on an observable game quantity, not a hand-picked strategic
judgment, and is fully annealed to 0 before phase 3 (league) starts.

## 3. Exploration

**Adaptive entropy controller** — LIVE **[rebuild: ❌ drop]** (`ENTROPY_ADAPTIVE=True`, league-phase-gated).
SAC-style automatic temperature: holds combined (action+duration) entropy at
`ENTROPY_TARGET_H=2.5` by nudging `ent_coef` up/down each update. Exploiters keep
their own two-stage scheduled coefficient instead (they must commit low to
best-respond). **Known coupling, not a bug**: it targets *combined* H, so anything
that reshapes the duration head (the hold bonus, when it was on) mildly self-dampens
through it — the clean fix would be targeting action entropy alone.

**Duration entropy: separate coefficient + floor** — LIVE. **[rebuild: ✅ keep, follows from FiGAR]**
`duration_entropy_coef = max(DURATION_ENTROPY_MULT * ec, DURATION_ENTROPY_COEF_FLOOR)`,
currently `DURATION_ENTROPY_MULT=1.5` (was `2.0` until today), `FLOOR=0.015`. A shared
coefficient let the duration head collapse (32/64-cycle holds sampled literally 0% of
the time under league pressure, specifically once an exploiter's action coefficient
crashed near its floor) — the multiplier gives it independently-tunable, always-
stronger pressure, and the floor stops it collapsing in lockstep with the action
coefficient during an exploiter's sharp late-phase commitment.
EDIT: when building, just keep the duration head entropy coefficient constant not based on ec.

**Param-noise exploration (exploiter behavior policy)** — LIVE, exploiter-only **[rebuild: DO NOT KEEP THIS FEATURE!!!, follows from the exploiter phase]**
(`EXPLOITER_PARAM_NOISE`). A perturbed copy of the actor (`behavior_actor`) is what
collects data; the clean actor is what trains — parameter-space noise instead of
purely action-space noise, AlphaStar-style. Sigma adapts per rollout
(`adapt_param_noise`/`resample_param_noise`).

**Mirror augmentation** — LIVE **[rebuild: ✅ keep]** (`MIRROR_MODE` = "sample" | "duplicate" | off).
Horizontal-symmetry data augmentation: negate x/vx, swap left/right key bits. Verified
exact (`mirror(expand(x)) == expand(mirror(x))`) for both the raw-window and
Fourier-expanded layouts. In recurrent mode, mirroring is applied per whole env
trajectory (never per-row) so a GRU sequence stays internally consistent.

**Spawn curriculum** — LIVE early **[rebuild: ✅ keep]**, permanent floor after
(`SPAWN_CURRICULUM_STEPS`, `SPAWN_CLOSE_PROB=0.40`). Measured 0.2% contact rate and 27m
median closest-approach with zero spawn variation — the outcome reward said nothing
about fighting. Ramps some episodes to spawn close early; a permanent 40% close-spawn
floor remains even once the curriculum fully opens, specifically so the KILL skill
isn't forgotten once far-apart spawns dominate (chasing an evader from max range is a
separate, unsolved pursuit problem this floor does not by itself teach).

## 4. Self-play / league

**AlphaStar-lite league (snapshots + PFSP)** — LIVE. **[rebuild: ✅ keep]** `league.py`. Past selves are
snapshotted periodically; opponent sampling is weighted toward whichever snapshot the
main currently *loses* to most (`_snap_weight`, PFSP over a rolling recent-games
window), not uniform — so training time concentrates on actual weaknesses.

**Exploiter phase (best-response training)** — LIVE. **[rebuild: ✅ keep]** Every
`EXPLOITER_INTERVAL` main episodes (mastery-gated: only when no live exploiter still
beats the main by `EXPLOITER_TRIGGER_WR`), main training freezes and a fresh exploiter
best-responds to the frozen main to find a hole. Graduates on a winrate gate (+ a
"sharpen tail") or a timeout, then joins the pool with PFSP weight like a snapshot.
**Historically only ~20% of exploiters reach the 70% gate** — most time out below it.
Exploiters get looser hyperparameters (bigger clip epsilon, higher LR, more epochs)
than the main, since they must diverge from it rather than stay near it.

**Staller exploiters** — LIVE **[rebuild: ✅ keep]** (`EXPLOITER_STALLER_PROB=0.25`). A quarter of exploiters
score a draw against the main as a WIN (turtle/stall), pressuring the main to learn to
force a kill rather than just avoid losing — the main's own gate can only be cleared by
beating a staller, which needs exactly the skill it otherwise lacks. The main's own
timeout against a graduated staller is scored 0 (neutral), not -1, so its gradient
points at closing out the kill rather than at merely avoiding the staller.

**Tree search / lookahead at train time** — BUILT **[rebuild: ❌ drop]**, currently disabled/not part of the
live config. `lookahead.py`, `single_process_collector.py`,
`experiment_search_finetune.py`, `eval_lookahead.py`, `train_search_augmented.py`. A
K-candidate rollout with real physics fork/restore (`EngineSim.get_full_state`/
`set_full_state`, `LagEnv.snapshot_state`/`restore_state`) used to pick the collected
action for a small fraction of decisions. Tried down to the simplest variant (K
candidates, commit immediately, no continuation, direct value comparison — nicknamed
"m0"). **Result: inconclusive** — didn't measurably help or hurt the trained policy,
cost real throughput (dropped steps/s), and was pulled back out of the live run rather
than kept as a permanent net negative. Left in the codebase since the physics
fork/restore plumbing (`EngineSim`, `LagEnv` snapshotting) is generically useful and
was a real, separate piece of engineering.

## 5. Infrastructure (not modeling decisions — listed for completeness, not asked about)

- **Fused backward pass** — one `(actor_loss + critic_loss).backward()` instead of two
  separate calls, sound only because the two losses touch disjoint parameter sets.
  Equivalence-tested (`test_fused_backward_equivalence.py`) parameter-for-parameter,
  not just loss values. Cut backward-call count in half on the hottest loop (~60-68%
  of wall clock).
- **MPS shape-padding** — torch-MPS caches a compiled graph per tensor shape forever;
  rows are trimmed/padded to a bounded set of shapes so a variable batch size doesn't
  leak memory without bound across a long run.
- **`--force-exploiter [--exploiter-type staller|combat|random]`** (`train.py`) —
  bypasses the natural mastery-gate trigger to start an exploiter immediately, for
  on-demand isolated testing (built for, and used throughout, today's debugging).
- **`SingleProcessCollector`** — a `VecCollector`-compatible, in-process (not
  multiprocess) collector, built specifically so a trainer subclass can get direct
  access to `LagEnv` objects for tree search's physics fork/restore, which the real
  multiprocess collector deliberately never exposes.

---

*Cross-reference: `ARCHITECTURE.md` (module-by-module code map, minGRU internals,
benchmarks), `CLAUDE.md` (the two hard constraints every reward-shaping feature above
was checked against), and the memory files under
`~/.claude/projects/.../memory/ppo7-*.md` for the full evidence trail behind each
DISABLED/contested item.*
