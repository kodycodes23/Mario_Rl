"""Procedural levels for generalization training.

Uses the same left-to-right Cursor as `level_authoring.py` so random maps keep a fair
landing buffer between obstacles (tight packing makes jumps physically unfair).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from game.levels.level_authoring import Cursor, build


def generate_level(
    seed: int,
    width: int = 55,
    height: int | None = None,
    n_pits: int | None = None,
    n_enemies: int | None = None,
    n_hazards: int | None = None,
    name: str | None = None,
) -> dict:
    del width  # width is a consequence of obstacle count + SAFE_BUFFER, not an input
    rng = np.random.default_rng(seed)
    height = int(height) if height is not None else int(rng.choice([9, 10]))
    n_pits = int(n_pits) if n_pits is not None else int(rng.integers(1, 4))
    n_enemies = int(n_enemies) if n_enemies is not None else int(rng.integers(1, 4))
    n_hazards = int(n_hazards) if n_hazards is not None else int(rng.integers(1, 3))

    kinds: list[str] = (
        ["enemy"] * n_enemies + ["pit"] * n_pits + ["hazard"] * n_hazards
    )
    rng.shuffle(kinds)

    cursor = Cursor()
    for kind in kinds:
        if kind == "enemy":
            cursor.enemy(width=int(rng.integers(4, 7)))
        elif kind == "pit":
            cursor.pit(width=int(rng.integers(2, 5)))
        else:
            cursor.hazard()

    return build(name or f"procgen_seed{seed}", cursor, height=height)


def save_level(level: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(level, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="game/levels/level_procgen.json")
    args = parser.parse_args()

    level = generate_level(seed=args.seed)
    save_level(level, args.out)
    print(f"wrote {args.out}")
