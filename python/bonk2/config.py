"""bonk2 — THE single config file. Every tunable for the v2 pipeline lives
here; no constants hide in ppo.py / train.py / league.py (the v1 mess this
rewrite exists to fix). Physics constants stay in bonk.config (owned by the
verified sim — do not duplicate them here).

v2 vs v1 deltas: HIDDEN [512,512] (was [256,256]), ACTION_REPEAT 2 (was 4),
GAMMA 0.995 per decision (was 0.99 — half the decision interval, so the
per-decision discount must rise to keep the same per-second horizon),
STATE_DIM 34 (was 30 — adds lastSeenHeavy + ticksSince per player), and an
explicit 30/60/10 current/PFSP/uniform opponent mix.
"""

# ── Environment ────────────────────────────────────────────────────────────────
TPS = 30
MAX_EPISODE_STEPS = 3000       # ticks (100 s); must match play3.mjs
ACTION_REPEAT = 2              # ticks per decision; must match play3.mjs
# Netcode model — must match how the real game feels to the deployed agent:
#   "rollback" (what bonk.io actually runs): your OWN inputs apply instantly;
#       the degradation is in your VIEW of the opponent — their last LAG ticks
#       of inputs haven't arrived, so you see a client-side prediction (their
#       t-LAG state extrapolated with held inputs) that is wrong exactly when
#       they change keys. play3.mjs observes this view natively, so training
#       with it is the sim-to-real bridge.
#   "delay": legacy v1 model — both seats' inputs apply LAG ticks late,
#       opponent state observed perfectly. Kept for A/B comparison.
NETCODE = "rollback"
INPUT_LAG = 5                  # view/apply lag in ticks (fixed fallback)
INPUT_LAG_RANDOM = True        # randomize lag per episode (network jitter)
INPUT_LAG_MIN = 2
INPUT_LAG_MAX = 6

# Sparse TERMINAL rewards only — the experiment's hard constraint. No shaping.
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
# Draw = timeout in practice (both-dead is rare). -0.9 makes clock-running
# nearly as bad as losing from BOTH sides of the scoreboard — with the short
# MAX_EPISODE_STEPS clock, the only profitable plan is forcing engagements.
DRAW_REWARD = -0.9

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
HIDDEN = [512, 512]
# Per decision (= 2 ticks). 0.995 -> 200-decision (13.3 s) credit horizon,
# matching the v1 run that worked (0.99 @ K=4). Lower values starve early-round
# actions of terminal credit: at 0.992 a win 500 decisions away is discounted
# to ~0.02 — invisible under sparse-terminal-only rewards.
GAMMA = 0.995
GAE_LAMBDA = 0.95
ACTOR_LR = 2e-4
CRITIC_LR = 3e-4
CLIP_EPS = 0.2
# 2 epochs, not 4: the update is the throughput bottleneck (~50x the FLOPs of
# collection), epochs scale it linearly, and PPO's clip makes passes 3-4
# mostly-saturated ratios. Measured 2x steps/s for a mild sample-efficiency
# cost (OpenAI Five ran ~1 epoch at this batch scale).
EPOCHS = 2
MINIBATCH = 16_384
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
NORMALIZE_ADV = True
CLIP_VALUE_LOSS = True
ROLLOUT_STEPS = 98_304         # learner decisions per PPO update
# Horizontal-mirror augmentation mode:
#   "sample"    - randomly mirror half the rows IN PLACE: same unique data and
#                 symmetry pressure at HALF the update rows of "duplicate"
#                 (2x faster updates; also halves the mirrored-logp ratio bias)
#   "duplicate" - v1 behavior: every row appears twice (original + mirrored)
#   "off"       - no augmentation
MIRROR_MODE = "sample"

# Entropy coefficient anneals linearly START -> END over the first
# ENTROPY_DECAY_EPISODES main episodes, then holds at END.
ENTROPY_COEF_START = 0.2
ENTROPY_COEF_END = 0.01
ENTROPY_DECAY_EPISODES = 1_500_000

# ── League ─────────────────────────────────────────────────────────────────────
# Main-phase opponent mix per episode:
#   OPP_CURRENT_PROB  -> mirror match vs the live current model (both seats train)
#   OPP_PFSP_PROB     -> pool member drawn by PFSP (weighted by main's loss rate)
#   OPP_UNIFORM_PROB  -> pool member drawn uniformly (coverage: keeps winrates
#                        fresh vs "solved" opponents)
# Must sum to 1. Empty pool falls back to current.
OPP_CURRENT_PROB = 0.30
OPP_PFSP_PROB = 0.60
OPP_UNIFORM_PROB = 0.10

SNAPSHOT_INTERVAL = 100_000    # main episodes between frozen self-snapshots
SNAPSHOT_BUFFER = 50
INITIAL_RATING = 1000.0
ELO_K = 16.0

# PFSP weight floors/priors (see league._snap_weight/_exp_weight).
PFSP_MIN_GAMES = 30            # games needed before a live loss rate is trusted
PFSP_SNAP_FLOOR = 0.1
PFSP_SNAP_PRIOR = 0.5
PFSP_EXP_FLOOR = 0.15
RECENT_CAP = 200               # rolling per-opponent result window

# Exploiter generation is adaptive, not on a fixed clock: spawn a fresh
# best-response once the main has absorbed its EXPLOITERS — none still beats
# it by EXPLOITER_TRIGGER_WR or more. Snapshots are excluded from this signal
# (a recent snapshot is ~the current policy, so it holds ~50% forever and
# would pin the max above any reachable threshold). MIN_INTERVAL stops
# thrashing right after one joins; MAX_INTERVAL is the diversity floor — a
# persistent, unpatchable hole can't freeze exploiter generation forever.
EXPLOITER_TRIGGER_WR = 0.45
EXPLOITER_MIN_INTERVAL = 30_000    # main episodes
EXPLOITER_MAX_INTERVAL = 300_000
EXPLOITER_TRIGGER_MIN_GAMES = 30

# Exploiter phase length is adaptive (winrate gate + sharpen tail + timeout).
EXPLOITER_MIN_EPISODES = 20_000
# 100k, not 150k: exploiter episodes come out of the main agent's training
# budget, and a phase that can't gate by 100k almost never gates by 150k —
# timed-out phases are the least valuable place to spend episodes.
EXPLOITER_MAX_EPISODES = 100_000
EXPLOITER_TARGET_WR = 0.7
EXPLOITER_EXTRA_EPISODES = 10_000
EXPLOITER_WR_WINDOW = 1000
EXPLOITER_WARM_START = True    # start from main's weights (False = from scratch)
EXPLOITER_POOL_MAX = 30

# Exploiter entropy: with WARM_START, EC_START must stay LOW — a high entropy
# bonus melts the transferred policy back to ~uniform within a few thousand
# episodes, silently converting the warm start into a cold one.
EXPLOITER_EC_START = 0.1
EXPLOITER_EC_END = 0.01
EXPLOITER_EC_DECAY_EPISODES = 20_000
