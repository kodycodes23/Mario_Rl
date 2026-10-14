"""World: owns entities, advances one fixed timestep per tick(action). Pure logic, no rendering.

This is the sole place game state lives and mutates. `env/mario_env.py` wraps this in the
Gymnasium API; the renderer only ever reads from a World instance, never writes to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game.constants import (
    ACTION_TO_INPUT,
    DT,
    REWARD_DEATH_PENALTY,
    REWARD_GOAL_BONUS,
    REWARD_POSITION_SCALE,
    REWARD_PROGRESS_SCALE,
    REWARD_STOMP,
    REWARD_TIME_PENALTY,
    STOMP_BOUNCE,
    Action,
    DeathCause,
)
from game.entities import Enemy, Goal, Hazard, Player
from game.physics import aabb_overlap
from game.tilemap import TileMap


@dataclass
class WorldState:
    """Snapshot of everything an observation/renderer needs. References into World, not copies."""

    player: Player
    enemies: list[Enemy]
    hazards: list[Hazard]
    goal: Goal
    tilemap: TileMap
    elapsed_steps: int
    elapsed_seconds: float
    max_x_reached: float


class World:
    def __init__(self, tilemap: TileMap):
        self.tilemap = tilemap
        self.rng: np.random.Generator = np.random.default_rng()
        self.player: Player | None = None
        self.enemies: list[Enemy] = []
        self.hazards: list[Hazard] = []
        self.goal: Goal | None = None
        self.elapsed_steps = 0
        self.max_x_reached = 0.0
        self._prev_jump_held = False

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> WorldState:
        self.rng = np.random.default_rng(seed)

        px, py = self.tilemap.player_spawn
        self.player = Player(x=px, y=py)
        self._prev_jump_held = False

        self.enemies = []
        self.hazards = []
        for spawn in self.tilemap.entity_spawns:
            if spawn.kind == "enemy":
                pmin = spawn.patrol_min_x if spawn.patrol_min_x is not None else spawn.x - 2.0
                pmax = spawn.patrol_max_x if spawn.patrol_max_x is not None else spawn.x + 2.0
                self.enemies.append(
                    Enemy(x=spawn.x, y=spawn.y, subtype=spawn.subtype,
                          patrol_min_x=pmin, patrol_max_x=pmax)
                )
            elif spawn.kind == "hazard":
                self.hazards.append(Hazard(x=spawn.x, y=spawn.y, subtype=spawn.subtype))

        gx, gy = self.tilemap.goal_pos
        self.goal = Goal(x=gx, y=gy)

        self.elapsed_steps = 0
        self.max_x_reached = self.player.x

        return self._snapshot()

    # ------------------------------------------------------------------- tick
    def tick(self, action: Action) -> tuple[WorldState, dict, bool, dict]:
        assert self.player is not None, "call reset() before tick()"

        move_left, move_right, jump_held = ACTION_TO_INPUT[Action(action)]
        jump_pressed = jump_held and not self._prev_jump_held
        self._prev_jump_held = jump_held

        prev_x = self.player.x
        self.player.step(DT, move_left, move_right, jump_held, jump_pressed, self.tilemap)
        for enemy in self.enemies:
            enemy.step(DT, self.tilemap, self.rng)

        self.elapsed_steps += 1

        terminated = False
        death_cause = DeathCause.NONE
        goal_reached = False
        stomps = 0

        player_box = self.player.aabb

        # fell into a pit (below the level)
        row_below = int(player_box.bottom)
        col_center = int(player_box.x + player_box.w / 2)
        if self.tilemap.is_pit(col_center, row_below):
            terminated = True
            death_cause = DeathCause.PIT

        # hazard contact
        if not terminated:
            for hazard in self.hazards:
                if aabb_overlap(player_box, hazard.aabb):
                    terminated = True
                    death_cause = DeathCause.HAZARD
                    break

        # enemy contact: falling onto an enemy from above stomps it (Mario bounce);
        # side or underside contact is still lethal.
        if not terminated:
            for enemy in self.enemies:
                if not enemy.alive or not aabb_overlap(player_box, enemy.aabb):
                    continue
                player_cy = player_box.y + player_box.h / 2
                enemy_cy = enemy.aabb.y + enemy.aabb.h / 2
                if self.player.vy > 0 and player_cy < enemy_cy:
                    enemy.alive = False
                    self.player.vy = -STOMP_BOUNCE
                    stomps += 1
                else:
                    terminated = True
                    death_cause = DeathCause.ENEMY
                    break

        # goal contact
        if not terminated and self.goal is not None and aabb_overlap(player_box, self.goal.aabb):
            terminated = True
            goal_reached = True
            death_cause = DeathCause.GOAL

        # --- reward components (raw; env/wrappers.py decides how to combine/weight these) ---
        prev_max_x = self.max_x_reached
        self.max_x_reached = max(self.max_x_reached, self.player.x)
        progress = (self.max_x_reached - prev_max_x) * REWARD_PROGRESS_SCALE
        position = (self.player.x - prev_x) * REWARD_POSITION_SCALE

        reward_components = {
            "progress": progress,
            "position": position,
            "time_penalty": REWARD_TIME_PENALTY,
            "goal_bonus": REWARD_GOAL_BONUS if goal_reached else 0.0,
            "death_penalty": REWARD_DEATH_PENALTY if (terminated and not goal_reached) else 0.0,
            "stomp": stomps * REWARD_STOMP,
        }

        goal_x, goal_y = self.tilemap.goal_pos
        total_dist = max(abs(goal_x - self.tilemap.player_spawn[0]), 1e-6)
        dist_remaining = max(goal_x - self.player.x, 0.0)
        x_progress_frac = float(np.clip(1.0 - dist_remaining / total_dist, 0.0, 1.0))

        info = {
            "death_cause": int(death_cause),
            "goal_reached": goal_reached,
            "x_progress_frac": x_progress_frac,
        }

        return self._snapshot(), reward_components, terminated, info

    # --------------------------------------------------------------- helpers
    def _snapshot(self) -> WorldState:
        return WorldState(
            player=self.player,
            enemies=self.enemies,
            hazards=self.hazards,
            goal=self.goal,
            tilemap=self.tilemap,
            elapsed_steps=self.elapsed_steps,
            elapsed_seconds=self.elapsed_steps * DT,
            max_x_reached=self.max_x_reached,
        )
