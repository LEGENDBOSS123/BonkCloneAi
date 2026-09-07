# Python mirror of the sim-relevant parts of src/rl2/config.mjs. Keep in sync —
# the whole point of this port is bit-level behavioral parity with the JS sim.

TPS = 30
DT = 1.0 / TPS
VELOCITY_ITERATIONS = 2
POSITION_ITERATIONS = 6

GRAVITY = 20.0

# Player (src/entities/player.mjs)
PLAYER_RADIUS = 1.0
MOVE_ACCEL = 12.0            # force applied per axis while a key is held
JUMP_SPEED = 10.0            # impulse = jumpSpeed * mass, upward
JUMP_VY_THRESHOLD = 4.0      # no jump if |vy| >= this
GROUND_MARGIN = 0.15         # ground ray reaches radius + margin below center
# Can a player treat another player as ground (i.e. jump off their head)?
# True  = JS-reference behavior (src/entities/player.mjs excludes only your own
#         collider, so an opponent under you counts as ground).
# False = players are not a floor; the ground ray passes through them.
# This is a deliberate DIVERGENCE from the JS sim — flip it back to True and
# re-run verify_replay.py if you need JS parity.
JUMP_OFF_PLAYERS = False
HEAVY_MAX = 1000.0
HEAVY_DRAIN = 10.0           # per tick while held
HEAVY_REGEN = 5.0            # per tick while released
HEAVY_EXTRA_MASS = 3.7       # additional mass at full heavyPower

# Ball fixture (src/physics.mjs addBall)
BALL_RESTITUTION = 0.95
BALL_FRICTION = 0.0

# Kill boundaries in bonk pixels (src/env.mjs), scaled by the map's ppm.
KILL_LINE_Y = 250.0
KILL_RADIUS = 850.0

# rl2 env (src/rl2/config.mjs)
MAX_EPISODE_STEPS = 8000
INPUT_LAG = 3
# Domain randomization: sample each episode's lag uniformly from
# [INPUT_LAG_MIN, INPUT_LAG_MAX]. Real bonk lag is variable (network + input
# pipeline), and a policy trained at one exact lag mistimes heavy in the real
# game; training across a range forces lag-robust timing. Set False to pin
# the lag at INPUT_LAG.
INPUT_LAG_RANDOM = True
INPUT_LAG_MIN = 2
INPUT_LAG_MAX = 4
ACTION_REPEAT = 4
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = -0.5

# Observation normalization (src/rl2/config.mjs obs)
POS_SCALE = 1 / 30
VEL_SCALE = 1 / 30
HEAVY_SCALE = 1 / 1000

NUM_ACTIONS = 18
STATE_DIM = 30
