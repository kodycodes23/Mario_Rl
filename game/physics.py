"""Gravity + AABB-vs-tilemap collision resolution. Pure functions, no state, no pygame.

Collision is resolved axis-separately (move+resolve X, then move+resolve Y) to avoid tunneling
and to get clean, classic-platformer wall/ground/ceiling behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

from game.constants import GRAVITY, TERMINAL_VELOCITY
from game.tilemap import TileMap


@dataclass
class AABB:
    x: float       # left edge, tile units
    y: float       # top edge, tile units
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h


def aabb_overlap(a: AABB, b: AABB) -> bool:
    return a.left < b.right and a.right > b.left and a.top < b.bottom and a.bottom > b.top


def apply_gravity(vy: float, dt: float) -> float:
    vy = vy + GRAVITY * dt
    return min(vy, TERMINAL_VELOCITY)


def _cols_overlapping(box: AABB) -> range:
    return range(int(box.left), int(box.right - 1e-9) + 1)


def _rows_overlapping(box: AABB) -> range:
    return range(int(box.top), int(box.bottom - 1e-9) + 1)


def move_and_collide_x(box: AABB, dx: float, tilemap: TileMap) -> tuple[float, bool]:
    """Moves box by dx, resolving solid-tile collisions on the X axis only.

    Returns (new_x, hit_wall).
    """
    new_x = box.x + dx
    probe = AABB(new_x, box.y, box.w, box.h)
    hit = False
    rows = _rows_overlapping(probe)
    if dx > 0:
        for col in _cols_overlapping(probe):
            for row in rows:
                if tilemap.is_solid(col, row):
                    new_x = min(new_x, col - box.w)
                    hit = True
    elif dx < 0:
        for col in _cols_overlapping(probe):
            for row in rows:
                if tilemap.is_solid(col, row):
                    new_x = max(new_x, col + 1)
                    hit = True
    return new_x, hit


def move_and_collide_y(box: AABB, dy: float, tilemap: TileMap) -> tuple[float, bool, bool]:
    """Moves box by dy, resolving solid-tile collisions on the Y axis only.

    Returns (new_y, hit_ground, hit_ceiling).
    """
    new_y = box.y + dy
    probe = AABB(box.x, new_y, box.w, box.h)
    hit_ground = False
    hit_ceiling = False
    cols = _cols_overlapping(probe)
    if dy > 0:
        for row in _rows_overlapping(probe):
            for col in cols:
                if tilemap.is_solid(col, row):
                    new_y = min(new_y, row - box.h)
                    hit_ground = True
    elif dy < 0:
        for row in _rows_overlapping(probe):
            for col in cols:
                if tilemap.is_solid(col, row):
                    new_y = max(new_y, row + 1)
                    hit_ceiling = True
    return new_y, hit_ground, hit_ceiling
