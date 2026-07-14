# RESEARCH.md — novelty analysis for a paper

Purpose: an honest map of which mechanisms in this project are **prior art** (so the
paper positions against them correctly) versus **candidate novel contributions**
(what you can claim). For each novel claim: the closest prior work, why it may be
novel, how to *substantiate* it with an ablation, and a caveat. **Do not overclaim
— reviewers punish it.** Hedge with "to our knowledge" unless you've done a lit
search. Novelty here is mostly at the level of *scheduling heuristics and a
controlled setup*, not a new learning algorithm — that's fine; frame the paper as a
systems/empirical study, not a new-optimizer paper.

---

## 1. System in one paragraph
Self-play RL agent for a Bonk.io-style 1v1 physics brawler, trained under two
deliberate constraints: **sparse terminal rewards only** (+1/−1/−0.5 at round end,
no shaping) and a **minimal, map-agnostic 30-D observation** (no hand-crafted
features). Training is an **AlphaStar-lite league** — a single main PPO agent,
periodic best-response *exploiters*, a pool of frozen past *snapshots*, and
PFSP-weighted opponent sampling — run **sequentially on one CPU machine**. A
verified Box2D port of the game physics enables fast headless training; **input-lag
domain randomization** bridges sim to the real browser game.

---

## 2. Prior art (NOT novel — position against these)
Be explicit about these in Related Work so novelty claims stand out.

- **League training / main + exploiters + snapshots** → AlphaStar (Vinyals et al.,
  2019). The overall architecture is theirs. You are running a *reduced* version.
- **PFSP (Prioritized Fictitious Self-Play)** — weighting opponents by winrate →
  AlphaStar. Weighting toward opponents you lose to is their `f_hard`.
- **PSRO** (best-response oracle + meta-game over a growing population) → Lanctot
  et al., 2017. Exploiters ≈ approximate best-response oracles.
- **Fictitious self-play / NFSP** → Heinrich et al., 2015; Heinrich & Silver, 2016.
- **Sparse terminal-reward self-play** → AlphaZero (Silver et al.). Win/loss-only is
  standard for board games; not novel on its own.
- **PPO, GAE, entropy regularization, clipped value loss** → Schulman et al. All
  standard.
- **Domain randomization for sim-to-real** → Tobin et al., 2017; OpenAI dactyl.
  Randomizing over a nuisance parameter to force robustness is a known template.
- **Action/observation delay in RL** → "delay-aware" / random-delay MDP literature
  (e.g., Ramstedt & Pal; Bouteiller et al., 2021). Delayed actions are studied.
- **ELO for tracking self-play** → standard; its pathologies (non-transitivity,
  cycling) are known folklore.

---

## 3. Candidate novel contributions (ranked by defensibility)

### 3.1 Mastery-triggered exploiter generation ★ strongest
**What:** a new best-response exploiter is spawned **not on a fixed clock** but when
the main has *mastered its current pool* — specifically when the highest winrate any
pool member holds against the main drops below a threshold `EXPLOITER_TRIGGER_WR`
(0.60), gated by a minimum spacing and a max-interval diversity fallback
(`_should_start_exploiter`, `_top_pool_winrate` in `train_ppo_mp.py`).

**Closest prior work:** AlphaStar adds agents to the league periodically and *graduates*
exploiters on a winrate gate; automatic-curriculum work (POET, Wang et al. 2019;
PAIRED, Dennis et al. 2020; "learning-progress" curricula) triggers new *tasks* when
the agent masters current ones. PFSP prioritizes *which* opponent to play.

**Why it may be novel:** to our knowledge, no league-training work ties the *creation
of a new adversary* to a **mastery signal computed as the max opponent-winrate over
the existing pool**. It imports the automatic-curriculum "advance when mastered" idea
into adversary *generation* (not task generation), in a competitive self-play league.
It's a small, clean, reportable heuristic — exactly the kind of "super small novel
thing" you asked for.

**How to substantiate (ablation):** fixed-interval exploiter spawning vs
mastery-triggered, matched on total compute. Metrics: (a) `eval_h2h` improvement rate
vs a fixed anchor, (b) fraction of exploiter compute "wasted" spawning while the main
still loses to existing exploiters, (c) time-to-patch a known hole. Claim survives
only if mastery-triggering improves sample efficiency or final strength.

**Caveat:** it's a scheduling heuristic, not a theorem. Frame as "an adaptive
trigger," report the ablation, don't claim optimality.

---

### 3.2 The diversity-floor fallback as a named failure-mode fix ★ supporting
**What:** pure mastery-triggering has a failure mode you identified: a **persistent,
unpatchable hole** (e.g. a juke the main can't fix) keeps the top pool winrate high
forever, so the trigger never fires and **exploiter generation freezes** — the league
stops discovering *other* holes. The `MAX_INTERVAL` fallback forces a spawn anyway.

**Why it may be novel / worth reporting:** this is a concrete, named pathology of
mastery-based adversary scheduling and a simple remedy. Even if 3.1 turns out to have
prior art, *characterizing this failure mode* (mastery-gating can deadlock on an
unlearnable hole) is a genuine, citable observation. Good papers report failure modes.

**Substantiate:** show a run where a hard hole exists; without the fallback, exploiter
count and pool diversity flatline; with it, new holes keep being found. This is a
clean qualitative figure.

---

### 3.3 Unified PFSP buffer with the current agent as a fixed-50% member ★ moderate
**What:** instead of AlphaStar's separate mixing probabilities over {self-play, past
selves, exploiters}, **one** PFSP distribution over the *entire* population, with the
live current model entered as an ordinary member at an assumed constant 50% winrate
(`_sample_opponent`, `CURRENT_PLAY_WEIGHT`). Snapshots/exploiters are weighted by the
main's **live** loss rate (rolling `recent` deques), with a small uniform floor.

**Closest prior work:** AlphaStar PFSP already recomputes winrates from recent games,
and already samples current/past/exploiter — but with *separate, hand-set* mixing
fractions and a two-level (kind → instance) structure. Reducing all of this to a
single winrate-prioritized draw where "play myself" is just a 50%-winrate entry is a
**reformulation/simplification**, and it removes a real bias (the two-level scheme
over-samples agents in smaller buckets purely for bucket size — see the derivation in
the code comments and our sampling test).

**Why it may be novel:** the *unification* (one flat PFSP pool incl. current-as-50%)
and the observation that two-level kind→instance sampling distorts per-agent
probability by bucket size. Modest, but a clean design point.

**Substantiate:** compare two-level mixing vs unified flat PFSP; measure whether a
single dangerous opponent in a large bucket gets adequate exposure; show the
bucket-size bias empirically. Also ablate `CURRENT_PLAY_WEIGHT`.

**Caveat:** this is close to prior PFSP; frame as a simplification with a measured
bias-removal, not a new algorithm.

---

### 3.4 Warm-start × entropy interaction finding ★ minor but real
**What:** when an exploiter **warm-starts** from the main's weights, a normal-strength
entropy bonus (ec≈0.3) **melts the transferred policy back to near-uniform**
(H → ln|A|) within a few thousand episodes, silently converting the warm start into a
de-facto cold start and wasting most of the phase. Fix: keep `EXPLOITER_EC_START` low
(0.05) so the inherited policy survives. (Documented in the constants block; we
observed the entropy-spike-then-recover curve directly in training logs.)

**Why worth reporting:** a concrete, measured interaction between two common tricks
(warm-starting a best-response + entropy regularization) with a clear diagnostic
signature (entropy spikes to max, winrate collapses to ~0, then slowly recovers).
Likely folklore but rarely written down; a small empirical "gotcha" contribution.

**Substantiate:** the entropy-vs-phase-episode curve at ec=0.3 vs ec=0.05, plus
episodes-to-90%-exploit for each. We have logs showing exactly this.

**Caveat:** minor; present as an empirical observation / practitioner note.

---

### 3.5 Input-latency domain randomization as the sim-to-real bridge ★ minor/applied
**What:** decisions apply `lag` ticks later, `lag` **randomized per episode**
(`env2.py` `LagEnv`, `INPUT_LAG_RANDOM`), so a policy trained headless survives the
real browser game's variable input+network latency without ever seeing the real game.

**Closest prior work:** domain randomization (Tobin et al.) and delay-aware RL. The
*technique* is known.

**Why possibly reportable:** the *application* — using per-episode action-latency
randomization specifically as the transfer mechanism from a headless physics sim to a
**deployed, human-facing browser game** — is an uncommon, concrete sim-to-real setting
(most DR work is robotics/vision). Report it as an applied result, not a new method,
ideally with a real-game transfer measurement (with/without lag DR).

**Caveat:** clearly not a novel method; novelty is only in the deployment target.

---

## 4. Methodological contributions (not algorithms, but paper-worthy)

### 4.1 The ELO-treadmill caution + fixed-anchor evaluation
**Claim:** ELO measured against a **rolling buffer of the agent's own recent selves**
is nearly uninformative about *absolute* progress — the yardstick rises with the
agent, so ELO plateaus whether or not it's improving. We instead use **fixed-anchor
head-to-head** (`eval_h2h.py`): current vs a frozen checkpoint from N episodes ago.
We empirically caught a run reading "flat ELO / positive PPO surrogate" that a
fixed-anchor test revealed as a **51.7% coin-flip (genuine plateau)**, and a later run
that read the same locally but was **80% over its 1M-old self**.

**Why worth including:** a crisp, reproducible demonstration that the two standard
"is it learning" signals (self-play ELO, PPO actor-loss) can *both* be flat/positive
while the agent is actually stuck — with a concrete fixed-anchor remedy. This is a
useful negative/methodological result. Known in spirit, but the paired demonstration
is clean.

### 4.2 PPO actor-loss is not a progress signal (worked demonstration)
We show (decomposition in the conversation/logs) that the logged actor loss is
~80% entropy bonus and its policy component is a small positive surrogate *every*
update regardless of whether the agent is improving. Good pedagogical/empirical note.

### 4.3 A single-machine, sequential AlphaStar-lite league as an accessible testbed
AlphaStar needed a large distributed population on TPUs with parallel exploiters. We
show a **sequential, single-CPU** league (one main, exploiter phases that *pause* main
training, shared-memory vectorized collection) that still exhibits the core dynamics
(cycling, exploiter hardening, PFSP anti-cycling). Value = **reproducibility/accessibility**:
a league you can run on a laptop. Report throughput and the compute-accounting tradeoff
(every exploiter episode is a main episode not trained — a constraint absent in
AlphaStar's parallel setup, and one that *motivates* the mastery-trigger in 3.1).

---

## 5. What is explicitly NOT novel (say so, to build credibility)
- Sparse terminal rewards; minimal observation (these are *ablation choices / a
  research question* — "can league self-play alone, with no shaping and no
  map-specific features, beat humans?" — not a method).
- PPO / GAE / league / PFSP / snapshots / mirror augmentation / ELO.
- The Box2D port and its verification (engineering + good reproducibility practice,
  not a research contribution — though worth a methods appendix).

---

## 6. Suggested framing for the paper
Title direction: *"Mastery-Triggered Adversary Generation in a Single-Machine
Self-Play League."* Lead with 3.1 + 3.2 (the trigger and its failure-mode fix) as the
primary contribution, support with 3.3 (unified PFSP) and 3.4 (warm-start/entropy),
and use §4 (evaluation methodology) as a secondary contribution. Position everything
under the controlled question of §5 (sparse + minimal-obs). Keep every novelty claim
paired with an ablation from §3, and hedge with "to our knowledge."

## 7. Honesty checklist before submission
- [ ] Lit-search each 3.x claim (curriculum learning, league training, PSRO variants,
      auto-curricula) — replace "to our knowledge" with citations or retract.
- [ ] Every novelty claim has a matched ablation with a number, not a vibe.
- [ ] Report the negative results (juke persistence, cycling) — they strengthen, not
      weaken, the paper.
- [ ] Distinguish "novel method" (none here) from "novel heuristic/combination/finding"
      (several) — claim only the latter.
