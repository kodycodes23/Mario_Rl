"""Cursor-based level authoring for level_3/4/5 (increasing-difficulty levels beyond the Core
level_1/level_2). Re-run to regenerate the JSON files from scratch:

    python -m game.levels.level_authoring

Why a cursor instead of hand-picked column numbers: the first draft of level_5 picked obstacle
positions by feel and got the spacing wrong in several places (gaps as low as 2-3 tiles between
an enemy's patrol zone and the next pit), which is exactly the failure mode documented in
writeup.md §3.2 -- a jump that clears one obstacle can carry straight into the next with no room
to land and re-jump, making the level unfair or outright impossible. A `Cursor` lays obstacles
out left-to-right and guarantees the validated minimum buffer by construction instead.

Two other bugs this file's comments exist to warn future edits away from:
- Every level needs >= ~16 tiles of clear runway from spawn before the first obstacle, or the
  player can die to the first enemy in under a second (the original level_1/level_2 bug).
- Decorative platforms must NOT be placed directly above an enemy's patrol zone: if a platform
  sits at a height the jump arc passes through, the player hits the ceiling mid-jump, the jump
  is cut short, and they fall back into the enemy below.

Every level produced here is verified beatable by
`tests/test_physics.py::test_all_levels_are_beatable` (a simple reactive jump-over-obstacles
script), across multiple seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

SAFE_BUFFER = 14  # tiles of clear ground between the end of one obstacle and the start of the
                  # next; the validated minimum is 12, this uses 14 for margin.
SPAWN_BUFFER = 16  # tiles of safe runway from spawn before the first obstacle.

LEVELS_DIR = Path(__file__).parent


class Cursor:
    """Lays out obstacles left-to-right, guaranteeing >= SAFE_BUFFER gap by construction."""

    def __init__(self, start: int = SPAWN_BUFFER):
        self.x = start
        self.pits: set[int] = set()
        self.hazards: list[int] = []
        self.enemies: list[tuple[float, int, int]] = []
        self.platforms: list[tuple[int, int, int]] = []

    def enemy(self, width: int = 5) -> "Cursor":
        start, end = self.x, self.x + width
        self.enemies.append(((start + end) / 2, start, end))
        self.x = end + SAFE_BUFFER
        return self

    def pit(self, width: int = 3) -> "Cursor":
        start, end = self.x, self.x + width
        self.pits.update(range(start, end))
        self.x = end + SAFE_BUFFER
        return self

    def hazard(self) -> "Cursor":
        col = self.x
        self.hazards.append(col)
        self.x = col + 1 + SAFE_BUFFER
        return self


def build(name: str, cursor: Cursor, height: int, goal_buffer: int = 6) -> dict:
    width = cursor.x + goal_buffer + 2
    ground_row = height - 1
    grid = [[0] * width for _ in range(height)]
    for c in range(width):
        if c not in cursor.pits:
            grid[ground_row][c] = 1
    for (row, cs, ce) in cursor.platforms:
        for c in range(max(0, cs), min(width, ce)):
            grid[row][c] = 2
    for hc in cursor.hazards:
        grid[ground_row - 1][hc] = 4
    goal_col = width - 2
    grid[ground_row - 1][goal_col] = 5

    entities = []
    for (cx, pmin, pmax) in cursor.enemies:
        entities.append({
            "kind": "enemy", "subtype": "walker", "x": float(cx - 0.4), "y": float(ground_row - 0.8),
            "patrol_min_x": float(pmin), "patrol_max_x": float(pmax),
        })
    for hc in cursor.hazards:
        entities.append({"kind": "hazard", "subtype": "spikes", "x": float(hc), "y": float(ground_row - 1)})

    return {"name": name, "tiles": grid, "player_spawn": [1.0, float(ground_row - 1)], "entities": entities}


def generate_all() -> list[dict]:
    # level_3 (medium): 2 enemies, 2 pits, 2 hazards -- same obstacle density as level_1, longer.
    c3 = Cursor()
    c3.enemy().pit().hazard().enemy().pit().hazard()
    level3 = build("level_3", c3, height=9)

    # level_4 (hard): 3 enemies, 3 pits, 2 hazards.
    c4 = Cursor()
    c4.enemy().pit().hazard().enemy().pit().enemy().hazard().pit()
    level4 = build("level_4", c4, height=10)

    # level_5 (hardest): 4 enemies, 4 pits, 3 hazards -- most obstacles, longest level, same
    # per-obstacle buffer as every other level (tighter spacing was tried and confirmed unfair).
    c5 = Cursor()
    c5.enemy().pit().hazard().pit().enemy().hazard().enemy().pit().hazard().pit().enemy()
    level5 = build("level_5", c5, height=10)

    return [level3, level4, level5]


if __name__ == "__main__":
    for level in generate_all():
        out_path = LEVELS_DIR / f"{level['name']}.json"
        with open(out_path, "w") as f:
            json.dump(level, f, indent=2)
        print(f"wrote {out_path} (width={len(level['tiles'][0])})")
