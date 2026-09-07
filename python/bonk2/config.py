"""bonk2 — THE single config file. Every tunable for the v2 pipeline lives
here; no constants hide in ppo.py / train.py / league.py (the v1 mess this
rewrite exists to fix). Physics constants stay in bonk.config (owned by the
verified sim — do not duplicate them here).
"""

# ── Physics backend ────────────────────────────────────────────────────────────
# "real"   -> bonk3: the bit-identical bonk.io engine (Rust, via ctypes).
#             Measured slightly FASTER than "legacy" at equal map complexity,
#             and it removes the sim-to-real physics gap entirely. Needs
#             python/bonk3/build.sh once, plus a decoded map (see MAP_NAME).
# "legacy" -> bonk.sim: the Box2D port of our own JS clone (~1e-5 vs the clone,
#             but the clone itself only approximates bonk).
# Switching backends changes the physics, so it invalidates checkpoints
# behaviourally (not by shape — STATE_DIM stays 34 either way).
ENGINE = "real"
MAP_NAME = "death"   # decoded stem, or a path to any MapData JSON
                                # ENGINE="legacy" uses --map / map1.json

# ── Environment ────────────────────────────────────────────────────────────────
TPS = 30
MAX_EPISODE_STEPS = 3000       # ticks (100 s); must match play3.mjs
ACTION_REPEAT = 2              # ticks per decision; must match play3.mjs
# Netcode model — must match how the real game feels to the deployed agent:
#   "rollback" (what bonk.io actually runs): your own inputs apply locally
#       without waiting for the server (SELF_LAG below covers the small
#       remaining browser/frame latency); the real degradation is in your VIEW
#       of the opponent — their last VIEW_LAG ticks of input haven't arrived,
#       so you see a client-side prediction that is wrong exactly when they
#       change keys. play3.mjs observes this view natively, so training with
#       it is the sim-to-real bridge.
#   "delay": legacy v1 model — both seats' inputs apply VIEW_LAG ticks late,
#       opponent state observed perfectly (no prediction). A/B only: with true
#       opponent velocity visible, feints/jukes cannot work in training.
NETCODE = "rollback"
# TWO INDEPENDENT LAGS, because the real game has both:
#
#  SELF_LAG  — your own keypress -> applied in the sim. Rollback applies your
#     input locally without waiting for the server, but it is NOT zero: the
#     browser key event lands mid-frame, gets quantized to a frame boundary,
#     and the deploy script sends on its own decision cadence. 0-4 ticks
#     (0-134 ms). This is what makes charge/contact timing (heavy!) land late
#     if you train with it at zero.
#
#  VIEW_LAG  — how stale your view of the OPPONENT is: their last VIEW_LAG
#     ticks of input haven't reached you, so you see a client-side prediction
#     that is wrong exactly when they change keys. Roughly half their ping +
#     half yours + frame quantization. 2-8 ticks (67-267 ms).
#
# Both are randomized per episode (jitter) and neither is observable, so the
# policy has to stay robust across the range.
LAG_RANDOM = True
SELF_LAG_MIN = 1
SELF_LAG_MAX = 4
SELF_LAG = 1                   # fixed fallback when LAG_RANDOM is False
VIEW_LAG_MIN = 3
VIEW_LAG_MAX = 8
VIEW_LAG = 5                   # fixed fallback when LAG_RANDOM is False

# The experiment's hard constraint, stated precisely: NO HARDCODED
# STATE-DEPENDENT RULE REWARDS. The line is about injected human strategy, not
# about state dependence per se. Forbidden is any hand-written rule that pays
# the agent for a game feature a person picked out — "closer to the opponent",
# "facing them", "heavy charged", "nearer the centre". Every one of those
# encodes somebody's theory of how bonk is played, and the whole experiment is
# whether the agent can find that out for itself.
#
# ALLOWED under this framing:
#   * a state-INDEPENDENT time cost — it says "sooner is better" and nothing
#     about bonk, exactly like gamma, which nobody calls reward shaping;
#   * a LEARNED intrinsic reward (e.g. RND novelty over the agent's own
#     observation block) — task-agnostic machinery that would be identical in
#     any environment and contains no bonk-specific knowledge;
#   * return-equivalent transformations such as RUDDER's redistribution, which
#     provably preserve the optimal policy.
#
# The test to apply: could a person who has never played bonk write this term?
# If yes it is allowed; if it required knowing what a good player does, it is not.
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
# Urgency: a win decays from WIN_REWARD to WIN_REWARD*(1-WIN_TIME_DECAY) over
# the episode clock, so a fast kill beats a slow one. 0.0 = flat (no urgency).
# This replaces discounting as the source of time pressure, and is strictly
# better for it: gamma scales wins AND losses by gamma^t, so it also makes
# dying LATER cheaper — paying the agent to stall. Decaying only the win keeps
# the pressure one-sided. (Discounting is itself a time-scaled terminal reward,
# so this is the same class of mechanism, just asymmetric and explicit.)
WIN_TIME_DECAY = 0.2
# Draw (= timeout in practice) must stay STRICTLY better than a loss. At -1.0
# a draw and a death are identical, which erases the only direct gradient for
# "don't fall off the edge" — in a knockoff game that survival ladder
# (die -1 < survive -0.6 < win +1) is the first skill the agent has to learn,
# and flattening it shows up as an agent that dies a lot. Keep it clearly
# worse than a win so running the clock is never a plan.
DRAW_REWARD = -0.5
# Per-DECISION time cost, applied to both seats every step (divided across the
# ACTION_REPEAT ticks inside a decision). State-independent: it says "sooner is
# better" and nothing about the game. Complements WIN_TIME_DECAY, which only
# pressures wins — this also pressures slow draws and slow losses.
#
# Sizing (the one real risk): an additive cost that accrues until the round
# ends means ending the round early is cheaper, so a losing agent can prefer to
# just die. That happens when p(win) < TIME_PENALTY * E[decisions remaining]/2.
# At 0.0002 over a full 1500-decision round the total cost is 0.30 and the
# give-up threshold is p<15% worst case, ~7% at a typical 700 decisions left —
# and it self-corrects, since the pressure shortens rounds, which shrinks
# E[remaining]. Raising this much above ~0.0003 starts making "give up when
# behind" genuinely rational. 0.0 disables it.
TIME_PENALTY = 0.0002

# Observation normalization.
POS_SCALE = 1 / 30
VEL_SCALE = 1 / 30
HEAVY_SCALE = 1 / 1000
# Ticks-since-heavy-seen normalizer: heavy regenerates at 5/tick, so 200 ticks
# = one full 0->1000 refill. With lastSeen scaled by HEAVY_SCALE, the net can
# estimate the true (hidden) meter as min(1, lastSeen + ticksSinceNorm).
HEAVY_SEEN_TICKS_NORM = 200

# 34-dim observation layout (see env.LagEnv.decision_state):
#   [ 0-11] self : x, y, vx, vy, heavyValue(masked by key), up, down, left,
#                  right, heavyKey, lastHeavySeen, ticksSinceSeen/200 (cap 1)
#   [12-23] opp  : same 12
#   [24-27] rel  : dx, dy, dvx, dvy
#   [28-32] pend : own last DECIDED action bits [up, down, left, right, heavy]
#   [33]    time : episode_steps / MAX_EPISODE_STEPS
STATE_DIM = 34
NUM_ACTIONS = 18               # 3 lr x 3 ud x 2 heavy joint actions

# Horizontal mirror of the obs (x/vx negate; left/right key bits swap).
MIRROR_NEGATE = [0, 2, 12, 14, 24, 26]        # self x,vx | opp x,vx | rel dx,dvx
MIRROR_SWAPS = [(7, 8), (19, 20), (30, 31)]   # left<->right: self, opp, pending

# ── PPO ────────────────────────────────────────────────────────────────────────
HIDDEN = [512, 512]            # smaller net (fresh run); ~2.5x faster updates
# gamma=1 (undiscounted). Urgency now comes from WIN_TIME_DECAY instead, which
# applies it to wins only; gamma would also discount the loss, rewarding the
# agent for postponing death. Undiscounted returns are safe here because
# episodes are hard-capped (MAX_EPISODE_STEPS) so returns stay bounded, and
# they give clean credit assignment — no exponential starvation of the early
# actions that set up a kill. Variance is handled by GAE_LAMBDA, not gamma.
GAMMA = 1.0
GAE_LAMBDA = 0.97             # 0.95 keeps advantage variance in check under gamma=1
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
CLIP_EPS = 0.2
# 4 epochs: extracts more learning per collected game (games are the currency
# in self-play). Epochs scale update cost linearly, but the smaller [256,256]
# net makes updates ~2.5x cheaper, so 4 is affordable here. Drop to 2 if you
# become wall-clock-bound rather than sample-bound.
EPOCHS = 4
MINIBATCH = 8192*2   # rows per minibatch (chunk size for the update loop)
VALUE_COEF = 0.5
# Auxiliary self-supervised task on the CRITIC: from the same trunk features it
# uses for the value, also predict this seat's own position change over the next
# decision (in METRES). Free supervision — the target is just the next
# observation — and it forces the critic's representation to encode dynamics
# instead of memorising position->value. Better value estimates mean
# lower-variance advantages, which is how it reaches the policy.
#
# NOTE: actor and critic are SEPARATE networks here, so this improves the
# critic's representation only; the actor benefits indirectly, via the
# advantages. Predicting the DELTA (not the absolute position) is the point —
# absolute position is already in the observation, so predicting it teaches
# nothing. The head is training-only and never ships: export_model takes the
# actor alone. 0.0 disables.
AUX_NEXT_POS_COEF = 0.25

# Prediction horizons, in DECISIONS (same unit as DEATH_SOON_DECISIONS, so
# h decisions = h * ACTION_REPEAT ticks). One output pair per horizon.
#
# Short horizons are nearly free — one decision ahead is almost pure inertia.
# The long ones are where the signal is: predicting where you'll be 60 decisions
# (~4 s at K=2) out requires the trunk to encode the map's collision geometry
# and your own momentum through it, which is exactly the structure a critic
# under sparse terminal rewards otherwise has no pressure to learn.
#
# Targets are stored as MEAN displacement per decision (metres/decision), i.e.
# the h-step delta divided by h. Raw h-step deltas grow ~linearly with h, so
# their squared error grows ~h^2 and the 60-step head would drown out the rest
# under one shared coefficient. Dividing by h puts every horizon on the same
# scale. h=1 is unchanged by this, so the first output is bit-identical in
# meaning to the single-horizon head this replaces.
AUX_POS_HORIZONS = (1, 15, 30, 60)

# ── Shared actor/critic trunk ──────────────────────────────────────────────────
# False: two independent MLPs (the v1/v2 default). The aux heads then hang off
#   the CRITIC's trunk, so they shape the critic's representation and reach the
#   policy only indirectly, through lower-variance advantages.
# True: ONE trunk feeding the policy head, the value head, and every aux head.
#   This is the point of the switch — the aux losses now pressure the very
#   features the actor acts on, which is what "auxiliary tasks speed up sparse-
#   reward learning" (UNREAL) actually relies on. It is also cheaper: act_batch
#   does one trunk pass instead of two.
#
# The cost is gradient interference: the value and aux losses now push on the
# policy's features, and a badly scaled VALUE_COEF can drown the policy
# gradient. Shared trunks are standard for discrete-action PPO and usually win
# there; separate nets tend to win on continuous control. Judge it with
# eval_h2h at matched episodes, not by the loss curves.
#
# The actor stays a plain MLP (trunk + policy head) in BOTH modes, so the
# league, export_model and play3.mjs are untouched either way. Checkpoints are
# not interchangeable across the switch and are guarded on load.
SHARED_TRUNK = True

# ── Trust region: clipping and/or KL ───────────────────────────────────────────
# CLIP_EPS (above) is always active. These add the KL side of PPO on top; all of
# them use the k3 estimator kl ~= (r-1) - log r, which is unbiased, much lower
# variance than -log r, and — unlike -log r — never goes negative.
#
# KL_COEF > 0: add beta*KL(old||new) to the actor loss (PPO-penalty, from the
#   original PPO paper's second variant). Off by default because it largely
#   duplicates what the clip already does; useful mainly if the clip alone is
#   letting updates run too far.
KL_COEF = 0.0
# KL_ADAPTIVE: move beta toward KL_TARGET using Schulman's 1.5x rule. Only
# meaningful when KL_COEF > 0.
KL_ADAPTIVE = True
KL_TARGET = 0.02
KL_COEF_MIN, KL_COEF_MAX = 1e-4, 10.0
# TARGET_KL: hard early-stop. If an epoch's mean KL exceeds this, the remaining
# epochs of THIS update are skipped. This is the cheap, standard safety net
# (SB3 exposes it, CleanRL runs it) and it only ever fires when the update was
# already stepping too far — so it is on by default, unlike the penalty above.
# Checked once per EPOCH, not per minibatch: the comparison forces a host sync,
# and a per-minibatch sync would stall the MPS pipeline. None disables it.
TARGET_KL = 0.03

# ── Recurrence (in-match adaptation) ───────────────────────────────────────────
# A feedforward policy on one frame CANNOT adapt to an opponent — nothing about
# what they did seconds ago reaches it. A GRU's hidden state is that memory, and
# it is also an implicit opponent embedding learned end-to-end (no separate
# intent head, no latent-collapse problem).
#
# Measured cost: ~1.9x wall-clock (the update roughly 2.3x, worker physics
# unaffected). Sample efficiency is the real unknown — recurrent PPO often needs
# MORE episodes, so judge it with eval_h2h at matched episode counts, not by
# wall-clock feel.
RECURRENT = False
# For memory to be worth learning, the same opponent must persist across several
# episodes with the hidden state carried over — otherwise there is nothing
# stable to adapt to and the recurrence learns nothing. This also lowers
# opponent diversity per unit time, so it trades against PFSP anti-cycling.
OPPONENT_HOLD_EPISODES = 4

# Auxiliary heads on the shared recurrent trunk. These are LOSSES, not rewards:
# the RL objective stays outcome-only; only the representation is pressured.
# Free targets, and each aims the hidden state at something specific.
AUX_OPP_ACTION_COEF = 0.10   # predict the opponent's NEXT action (18-way).
                             # The opponent-modelling signal — to predict them
                             # the state must encode who they are. Most directly
                             # aimed at reading jukes/feints.
AUX_DEATH_COEF = 0.15        # predict "this seat dies within N decisions".
                             # Densifies an otherwise almost gradient-free
                             # critic under sparse terminal rewards.
DEATH_SOON_DECISIONS = 15    # horizon for the death head (~1 s at K=2)
MAX_GRAD_NORM = 0.5
NORMALIZE_ADV = True
CLIP_VALUE_LOSS = True
# Learner decisions per PPO update. 49152 / ~250 decisions per episode ≈ 195
# episodes of terminal-reward signal per update — comfortably above the ~130
# (32k) floor where win/loss outcome noise starts leaking into the advantages,
# without buying stability the run doesn't need.
ROLLOUT_STEPS = 16_384*3         # = 49_152
# Horizontal-mirror augmentation mode:
#   "sample"    - randomly mirror half the rows IN PLACE: same unique data and
#                 symmetry pressure at HALF the update rows of "duplicate"
#                 (2x faster updates; also halves the mirrored-logp ratio bias)
#   "duplicate" - v1 behavior: every row appears twice (original + mirrored)
#   "off"       - no augmentation
MIRROR_MODE = "duplicate"

# Entropy coefficient anneals START -> END (see ENTROPY_DECAY_MODE) over the
# first ENTROPY_DECAY_EPISODES main episodes, then holds at END.
ENTROPY_COEF_START = 0.1
ENTROPY_COEF_END = 0.01      # moderate floor; lower (0.01) for a sharper
                             # policy, raise if it reads as predictable to humans
ENTROPY_DECAY_EPISODES = 1_000_000
# "linear" (constant decrease/ep) or "exponential" (constant ratio/ep — drops
# fast early, long low tail, so the policy sharpens sooner). Faster to commit,
# but a shorter high-entropy window raises the risk of locking into a basin
# (e.g. the rush/juke) before the league can pressure it out.
ENTROPY_DECAY_MODE = "exponential"

# ── League ─────────────────────────────────────────────────────────────────────
# Main-phase opponent mix per episode:
#   OPP_CURRENT_PROB  -> mirror match vs the live current model (both seats train)
#   OPP_PFSP_PROB     -> pool member drawn by PFSP (weighted by main's loss rate)
#   OPP_UNIFORM_PROB  -> pool member drawn uniformly (coverage: keeps winrates
#                        fresh vs "solved" opponents)
# Must sum to 1. Empty pool falls back to current.
OPP_CURRENT_PROB = 0.3
OPP_PFSP_PROB = 0.6
OPP_UNIFORM_PROB = 0.10

SNAPSHOT_INTERVAL = 150_000    # main episodes between frozen self-snapshots
SNAPSHOT_BUFFER = 100
INITIAL_RATING = 1000.0
ELO_K = 16.0

# PFSP weight floors/priors (see league._snap_weight/_exp_weight).
PFSP_MIN_GAMES = 50            # games needed before a live loss rate is trusted
PFSP_SNAP_FLOOR = 0.1
PFSP_SNAP_PRIOR = 0.5
PFSP_EXP_FLOOR = 0.15
# Weight = max(loss_rate, floor) ** PFSP_POWER. 1.0 = plain loss-rate (linear).
# Higher concentrates play on the few opponents that actually beat the main —
# the anti-cycling lever. With a ~150-agent pool, linear weighting dilutes a
# dangerous opponent to ~5% of episodes (its 0.76 weight competes with 148
# floors of 0.1); squaring lifts that to ~25%, because floors fall to 0.01
# while the threat stays at 0.58. This is AlphaStar's f_hard prioritization.
PFSP_POWER = 2.0
RECENT_CAP = 200               # rolling per-opponent result window

# Exploiter generation is adaptive, not on a fixed clock: spawn a fresh
# best-response once the main has absorbed its EXPLOITERS — none still beats
# it by EXPLOITER_TRIGGER_WR or more. Snapshots are excluded from this signal
# (a recent snapshot is ~the current policy, so it holds ~50% forever and
# would pin the max above any reachable threshold). MIN_INTERVAL stops
# thrashing right after one joins; MAX_INTERVAL is the diversity floor — a
# persistent, unpatchable hole can't freeze exploiter generation forever.
EXPLOITER_TRIGGER_WR = 0.5
# 150k, not 80k: exploiter phases PAUSE main training, so this ratio decides
# how much of the run the shipped agent actually gets. At 80k against a
# 10k-100k phase the main saw only 44-67% of episodes; at 150k it's 60-94%.
EXPLOITER_MIN_INTERVAL = 100_000   # main episodes
EXPLOITER_MAX_INTERVAL = 200_000
EXPLOITER_TRIGGER_MIN_GAMES = 50

# Exploiter phase length is adaptive (winrate gate + sharpen tail + timeout).
EXPLOITER_MIN_EPISODES = 10_000
# 100k timeout: exploiter episodes pause main training, and a phase that can't
# gate by 100k almost never gates by 150k — timed-out phases are the least
# valuable place to spend episodes.
EXPLOITER_MAX_EPISODES = 100_000
# 0.72, not 0.85: an exploiter graduates once it beats the frozen main by this
# margin. 0.85 is so strict that most holes never reach it, so phases grind to
# the 100k timeout (max cost). 0.72 gates as soon as a *real* hole is found and
# hands the budget back to the main.
EXPLOITER_TARGET_WR = 0.72
EXPLOITER_EXTRA_EPISODES = 5_000
EXPLOITER_WR_WINDOW = 1500
EXPLOITER_WARM_START = True    # start from main's weights (False = from scratch)
EXPLOITER_POOL_MAX = 50

# Exploiter entropy: with WARM_START, EC_START must stay LOW — a high entropy
# bonus melts the transferred policy back to ~uniform within a few thousand
# episodes, silently converting the warm start into a cold one.
EXPLOITER_EC_START = 0.06
EXPLOITER_EC_END = 0.01
EXPLOITER_EC_DECAY_EPISODES = 15_000
