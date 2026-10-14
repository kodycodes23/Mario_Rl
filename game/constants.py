"""Physics constants, tile size, timestep, and action enum.

All units are in tiles (position/velocity) unless noted otherwise. The sim runs at a fixed
timestep regardless of wall-clock rendering speed, so these constants define "feel" once and
for all -- the renderer only ever reads state, it never changes physics.
"""
from __future__ import annotations

from enum import IntEnum

# --- Timestep -----------------------------------------------------------------
FPS = 60
DT = 1.0 / FPS

# --- Tile / world geometry -----------------------------------------------------
TILE_SIZE = 16  # pixels per tile, used only by the renderer for scaling

# --- Player physics (units: tiles / second, tiles / second^2) ------------------
GRAVITY = 45.0                # downward acceleration
TERMINAL_VELOCITY = 22.0      # max fall speed
RUN_ACCEL = 55.0              # horizontal acceleration while an input is held
RUN_DECEL = 65.0              # horizontal deceleration ("friction") with no input
MAX_RUN_SPEED = 8.5           # max horizontal speed
JUMP_IMPULSE = 15.5           # upward velocity applied on jump takeoff
JUMP_CUT_MULTIPLIER = 0.45    # multiply upward velocity by this if jump released early
MAX_COYOTE_TIME = 0.08        # seconds after leaving ground a jump still registers
PLAYER_W = 0.8                # player AABB width in tiles
PLAYER_H = 0.95               # player AABB height in tiles

# --- Enemy physics --------------------------------------------------------------
ENEMY_SPEED = 2.0
ENEMY_W = 0.8
ENEMY_H = 0.8
ENEMY_JITTER_STD = 0.05       # patrol speed jitter (seeded RNG only)
STOMP_BOUNCE = 10.0           # upward vy after a successful stomp (Mario bounce)

# --- Episode --------------------------------------------------------------------
MAX_EPISODE_SECONDS = 40.0
MAX_EPISODE_STEPS = int(MAX_EPISODE_SECONDS * FPS)

# --- Rewards (baseline shaped config; see configs/*.yaml for overrides) ---------
# Tuned so that idling for a full episode (REWARD_TIME_PENALTY * MAX_EPISODE_STEPS = -48) costs
# MORE than several deaths' worth of REWARD_DEATH_PENALTY. An earlier pass used -0.01/-50, under
# which idling out a full episode (-24) was cheaper than roughly one 50%-risk attempt at the
# goal -- PPO converged on a real, reproducible local optimum: advance partway (banking the
# progress reward, which uses a running high-water mark), then retreat to safety and coast to
# timeout rather than risk the final push. See writeup.md for the full trained-vs-random
# evidence. This rebalance is the fix: idling is now clearly worse than attempting the level.
REWARD_PROGRESS_SCALE = 1.0     # multiplier on +delta-x-toward-goal each step
REWARD_POSITION_SCALE = 0.3     # multiplier on current-x delta (negative if retreating)
REWARD_TIME_PENALTY = -0.02     # per-step penalty
REWARD_GOAL_BONUS = 100.0
REWARD_DEATH_PENALTY = -15.0
REWARD_STOMP = 5.0              # bonus for stomping an enemy from above

# --- Tile IDs --------------------------------------------------------------------
class Tile(IntEnum):
    EMPTY = 0
    GROUND = 1
    BRICK = 2
    PIT = 3       # visually empty but marks "kills on fall through" region below level
    HAZARD = 4    # spikes / lava - kills on contact
    GOAL = 5


SOLID_TILES = frozenset({Tile.GROUND, Tile.BRICK})
LETHAL_TILES = frozenset({Tile.HAZARD})


# --- Actions -----------------------------------------------------------------
class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


ACTION_TO_INPUT = {
    # (move_left, move_right, jump_held)
    Action.NOOP: (False, False, False),
    Action.LEFT: (True, False, False),
    Action.RIGHT: (False, True, False),
    Action.JUMP: (False, False, True),
    Action.LEFT_JUMP: (True, False, True),
    Action.RIGHT_JUMP: (False, True, True),
}

N_ACTIONS = len(Action)

# --- Death causes (for info dict / death-cause breakdown logging) ---------------
class DeathCause(IntEnum):
    NONE = 0
    PIT = 1
    ENEMY = 2
    HAZARD = 3
    TIMEOUT = 4
    GOAL = 5  # not a death, but reuses the same "episode end reason" slot
