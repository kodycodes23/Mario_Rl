"""Level loading from JSON and tile collision queries. No pygame here."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from game.constants import LETHAL_TILES, SOLID_TILES, Tile


@dataclass
class EntitySpawn:
    kind: str                 # "enemy" | "hazard"
    subtype: str              # e.g. "walker", "spikes"
    x: float
    y: float
    patrol_min_x: float | None = None
    patrol_max_x: float | None = None


@dataclass
class TileMap:
    tiles: np.ndarray            # shape (height, width), dtype=int, tiles[row, col]
    width: int
    height: int
    player_spawn: tuple[float, float]
    goal_pos: tuple[float, float]
    entity_spawns: list[EntitySpawn] = field(default_factory=list)
    name: str = "unnamed"

    @classmethod
    def from_dict(cls, data: dict, name: str = "unnamed") -> "TileMap":
        grid = data["tiles"]
        height = len(grid)
        width = len(grid[0])
        tiles = np.array(grid, dtype=np.int32)
        assert tiles.shape == (height, width), "ragged tile grid"

        goal_cells = np.argwhere(tiles == Tile.GOAL)
        assert len(goal_cells) >= 1, "level must contain exactly one GOAL tile"
        goal_row, goal_col = goal_cells[0]
        goal_pos = (float(goal_col), float(goal_row))

        spawn = data.get("player_spawn", None)
        if spawn is None:
            player_spawn = (1.0, 1.0)
        else:
            player_spawn = (float(spawn[0]), float(spawn[1]))

        entity_spawns = []
        for e in data.get("entities", []):
            entity_spawns.append(
                EntitySpawn(
                    kind=e["kind"],
                    subtype=e.get("subtype", "default"),
                    x=float(e["x"]),
                    y=float(e["y"]),
                    patrol_min_x=e.get("patrol_min_x"),
                    patrol_max_x=e.get("patrol_max_x"),
                )
            )

        return cls(
            tiles=tiles,
            width=width,
            height=height,
            player_spawn=player_spawn,
            goal_pos=goal_pos,
            entity_spawns=entity_spawns,
            name=data.get("name", name),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TileMap":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, name=data.get("name", path.stem))

    def tile_at(self, col: int, row: int) -> int:
        """Returns Tile.GROUND for out-of-bounds sides, PIT for below the map (falls to death),
        and EMPTY above the map."""
        if row >= self.height:
            return int(Tile.PIT)
        if row < 0:
            return int(Tile.EMPTY)
        if col < 0 or col >= self.width:
            return int(Tile.GROUND)
        return int(self.tiles[row, col])

    def is_solid(self, col: int, row: int) -> bool:
        return self.tile_at(col, row) in SOLID_TILES

    def is_lethal(self, col: int, row: int) -> bool:
        return self.tile_at(col, row) in LETHAL_TILES

    def is_pit(self, col: int, row: int) -> bool:
        return self.tile_at(col, row) == Tile.PIT

    def local_window(self, cx: float, cy: float, w: int, h: int) -> np.ndarray:
        """Returns a (h, w) int array of tile ids centered on the (cx, cy) tile-space position.

        Used to build the observation's local tile window feature.
        """
        col = int(round(cx))
        row = int(round(cy))
        half_w, half_h = w // 2, h // 2
        out = np.zeros((h, w), dtype=np.int32)
        for r_off in range(-half_h, h - half_h):
            for c_off in range(-half_w, w - half_w):
                out[r_off + half_h, c_off + half_w] = self.tile_at(col + c_off, row + r_off)
        return out
