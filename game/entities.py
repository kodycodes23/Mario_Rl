"""Player, Enemy, Hazard, Goal entity classes. Pure logic, no rendering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game.constants import (
    ENEMY_H,
    ENEMY_JITTER_STD,
    ENEMY_SPEED,
    ENEMY_W,
    JUMP_CUT_MULTIPLIER,
    JUMP_IMPULSE,
    MAX_COYOTE_TIME,
    MAX_RUN_SPEED,
    PLAYER_H,
    PLAYER_W,
    RUN_ACCEL,
    RUN_DECEL,
)
from game.physics import AABB, apply_gravity, move_and_collide_x, move_and_collide_y
from game.tilemap import TileMap


@dataclass
class Player:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    facing_right: bool = True
    time_since_ground: float = 999.0
    just_landed: bool = False   # set true for one tick on landing (for renderer squash/stretch)
    just_jumped: bool = False

    @property
    def aabb(self) -> AABB:
        return AABB(self.x, self.y, PLAYER_W, PLAYER_H)

    def step(self, dt: float, move_left: bool, move_right: bool, jump_held: bool,
              jump_pressed: bool, tilemap: TileMap) -> None:
        self.just_landed = False
        self.just_jumped = False

        # --- horizontal ---
        if move_left and not move_right:
            self.vx -= RUN_ACCEL * dt
            self.facing_right = False
        elif move_right and not move_left:
            self.vx += RUN_ACCEL * dt
            self.facing_right = True
        else:
            if self.vx > 0:
                self.vx = max(0.0, self.vx - RUN_DECEL * dt)
            elif self.vx < 0:
                self.vx = min(0.0, self.vx + RUN_DECEL * dt)
        self.vx = float(np.clip(self.vx, -MAX_RUN_SPEED, MAX_RUN_SPEED))

        # --- jump takeoff (with coyote time) ---
        can_jump = self.on_ground or self.time_since_ground <= MAX_COYOTE_TIME
        if jump_pressed and can_jump:
            self.vy = -JUMP_IMPULSE
            self.on_ground = False
            self.time_since_ground = 999.0
            self.just_jumped = True
        elif not jump_held and self.vy < 0:
            # variable jump height: cut upward velocity if the button is released early
            self.vy *= JUMP_CUT_MULTIPLIER

        # --- gravity ---
        self.vy = apply_gravity(self.vy, dt)

        # --- move + resolve, X then Y ---
        box = self.aabb
        new_x, hit_wall = move_and_collide_x(box, self.vx * dt, tilemap)
        if hit_wall:
            self.vx = 0.0
        self.x = new_x

        box = self.aabb
        new_y, hit_ground, hit_ceiling = move_and_collide_y(box, self.vy * dt, tilemap)
        was_airborne = not self.on_ground
        if hit_ground:
            if was_airborne and self.vy > 2.0:
                self.just_landed = True
            self.vy = 0.0
            self.on_ground = True
            self.time_since_ground = 0.0
        else:
            self.on_ground = False
            self.time_since_ground += dt
        if hit_ceiling:
            self.vy = max(self.vy, 0.0)
        self.y = new_y


@dataclass
class Enemy:
    x: float
    y: float
    subtype: str = "walker"
    patrol_min_x: float = 0.0
    patrol_max_x: float = 0.0
    vx: float = -ENEMY_SPEED
    vy: float = 0.0
    alive: bool = True

    @property
    def aabb(self) -> AABB:
        return AABB(self.x, self.y, ENEMY_W, ENEMY_H)

    def step(self, dt: float, tilemap: TileMap, rng: np.random.Generator) -> None:
        if not self.alive:
            return
        jitter = 1.0 + float(rng.normal(0.0, ENEMY_JITTER_STD))
        self.x += self.vx * jitter * dt

        if self.patrol_max_x > self.patrol_min_x:
            if self.x <= self.patrol_min_x:
                self.x = self.patrol_min_x
                self.vx = abs(self.vx)
            elif self.x >= self.patrol_max_x:
                self.x = self.patrol_max_x
                self.vx = -abs(self.vx)

        self.vy = apply_gravity(self.vy, dt)
        box = self.aabb
        new_y, hit_ground, _ = move_and_collide_y(box, self.vy * dt, tilemap)
        if hit_ground:
            self.vy = 0.0
        self.y = new_y


@dataclass
class Hazard:
    x: float
    y: float
    w: float = 1.0
    h: float = 1.0
    subtype: str = "spikes"

    @property
    def aabb(self) -> AABB:
        return AABB(self.x, self.y, self.w, self.h)


@dataclass
class Goal:
    x: float
    y: float
    w: float = 1.0
    h: float = 1.0

    @property
    def aabb(self) -> AABB:
        return AABB(self.x, self.y, self.w, self.h)
