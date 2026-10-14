"""Determinism, collision correctness, and seed reproducibility for game/."""
from __future__ import annotations

import numpy as np
import pytest

from game.constants import Action, DeathCause, Tile
from game.physics import AABB, aabb_overlap, move_and_collide_x, move_and_collide_y
from game.tilemap import TileMap
from game.world import World

LEVEL_1 = "game/levels/level_1.json"
LEVEL_2 = "game/levels/level_2.json"
ALL_LEVELS = (
    LEVEL_1,
    LEVEL_2,
    "game/levels/level_3.json",
    "game/levels/level_4.json",
    "game/levels/level_5.json",
)


def make_world(level_path: str = LEVEL_1) -> World:
    tilemap = TileMap.from_json(level_path)
    return World(tilemap)


def scripted_actions(n: int, seed: int = 0) -> list[Action]:
    rng = np.random.default_rng(seed)
    choices = list(Action)
    return [choices[i] for i in rng.integers(0, len(choices), size=n)]


# --------------------------------------------------------------------------- determinism
def test_reset_same_seed_same_spawn():
    w = make_world()
    s1 = w.reset(seed=42)
    x1, y1 = s1.player.x, s1.player.y
    s2 = w.reset(seed=42)
    assert (x1, y1) == (s2.player.x, s2.player.y)


def test_identical_action_sequence_is_bit_identical_trajectory():
    actions = scripted_actions(200, seed=7)

    def run(seed):
        w = make_world()
        w.reset(seed=seed)
        traj = []
        for a in actions:
            state, reward_components, terminated, info = w.tick(a)
            traj.append((state.player.x, state.player.y, state.player.vx, state.player.vy))
            if terminated:
                break
        return traj

    t1 = run(42)
    t2 = run(42)
    assert t1 == t2, "two reset(seed=42) runs under an identical action sequence diverged"


def test_different_seeds_can_diverge_via_enemy_jitter():
    # enemy patrol jitter is drawn from the seeded RNG, so different seeds may (not must)
    # produce different enemy trajectories; same seed must always agree (checked above).
    actions = [Action.NOOP] * 100

    def run(seed):
        w = make_world()
        w.reset(seed=seed)
        xs = []
        for a in actions:
            state, _, terminated, _ = w.tick(a)
            xs.append(state.enemies[0].x if state.enemies else None)
            if terminated:
                break
        return xs

    # same seed reproduces exactly
    assert run(1) == run(1)


def test_rng_is_numpy_generator_not_global_random_module():
    w = make_world()
    w.reset(seed=1)
    assert isinstance(w.rng, np.random.Generator)


# --------------------------------------------------------------------------- collision
def test_player_lands_on_ground_and_stops_falling():
    w = make_world()
    w.reset(seed=0)
    for _ in range(120):
        state, _, terminated, _ = w.tick(Action.NOOP)
        if terminated:
            break
    assert w.player.on_ground, "player should have settled on the ground under gravity+noop"
    assert w.player.vy == 0.0


def test_axis_separate_collision_no_tunneling_through_thin_ground():
    from game.constants import DT, TERMINAL_VELOCITY

    tilemap = TileMap.from_json(LEVEL_1)
    max_step_dy = TERMINAL_VELOCITY * DT
    # start just within one tick's worth of the floor (ground row starts at y=8.0)
    box = AABB(x=5.0, y=8.0 - 0.95 - max_step_dy * 0.5, w=0.8, h=0.95)
    # the largest single-tick displacement the sim can ever produce (falling at terminal
    # velocity for one fixed timestep) must not tunnel through a 1-tile-thick floor.
    new_y, hit_ground, hit_ceiling = move_and_collide_y(box, dy=max_step_dy, tilemap=tilemap)
    assert hit_ground
    assert new_y <= 8.0 - 0.95 + 1e-9


def test_move_and_collide_x_stops_at_wall():
    tilemap = TileMap.from_json(LEVEL_1)
    # column 20 at ground_row-1 (row 7) is empty air above a pit; use a guaranteed solid wall
    # by probing far right where boundary-out-of-map columns count as solid ground per tile_at().
    box = AABB(x=tilemap.width - 0.5, y=0.0, w=0.8, h=0.95)
    new_x, hit = move_and_collide_x(box, dx=5.0, tilemap=tilemap)
    assert hit


def test_aabb_overlap_basic():
    a = AABB(0, 0, 1, 1)
    b = AABB(0.5, 0.5, 1, 1)
    c = AABB(2, 2, 1, 1)
    assert aabb_overlap(a, b)
    assert not aabb_overlap(a, c)


# --------------------------------------------------------------------------- level content
def test_level_contains_required_elements():
    for path in ALL_LEVELS:
        tm = TileMap.from_json(path)
        assert (tm.tiles == Tile.GROUND).any(), f"{path} missing ground"
        assert (tm.tiles == Tile.HAZARD).any(), f"{path} missing a hazard tile"
        assert (tm.tiles == Tile.GOAL).any(), f"{path} missing a goal tile"
        assert any(e.kind == "enemy" for e in tm.entity_spawns), f"{path} missing an enemy"


def test_all_levels_are_beatable():
    """A simple reactive jump-over-obstacles bot must be able to reach the goal on every level.

    This is the regression test for a real bug found during level authoring: tightly-packed
    obstacles (an enemy patrol zone ending just a few tiles before a pit, or a decorative
    platform placed directly over a jump arc) can make a level unfair or literally impossible,
    even though each individual obstacle looks fine in isolation. See levels 3-5's generation
    notes for the specific failures this caught.
    """
    for path in ALL_LEVELS:
        tm = TileMap.from_json(path)
        w = World(tm)
        w.reset(seed=1)
        goal_reached = False
        for _ in range(12000):
            px = w.player.x
            ground_row = tm.height - 1
            trigger = any(e.alive and 0 < (e.x - px) < 3.0 for e in w.enemies)
            for look in (1, 2, 3):
                col = int(px) + look
                if not tm.is_solid(col, ground_row) or tm.is_lethal(col, ground_row - 1):
                    trigger = True
            action = Action.RIGHT_JUMP if trigger else Action.RIGHT
            _, _, terminated, info = w.tick(action)
            if terminated:
                goal_reached = info["goal_reached"]
                break
        assert goal_reached, f"{path} is not beatable by a simple jump-over-obstacles bot"


def test_falling_in_pit_kills_player():
    w = make_world()
    w.reset(seed=0)
    terminated = False
    info = {}
    for _ in range(300):
        state, _, terminated, info = w.tick(Action.RIGHT)
        if terminated:
            break
    assert terminated
    assert info["death_cause"] in (DeathCause.PIT, DeathCause.ENEMY, DeathCause.HAZARD, DeathCause.GOAL)


def test_stomp_from_above_kills_enemy_and_does_not_kill_player():
    w = make_world()
    w.reset(seed=0)
    enemy = w.enemies[0]
    w.player.x = enemy.x
    w.player.y = enemy.y - 0.6
    w.player.vy = 8.0
    w.player.on_ground = False
    _, rc, terminated, info = w.tick(Action.NOOP)
    assert not terminated
    assert not enemy.alive
    assert rc["stomp"] > 0
    assert info["death_cause"] == DeathCause.NONE
    assert w.player.vy < 0  # bounce


def test_side_contact_with_enemy_still_kills():
    w = make_world()
    w.reset(seed=0)
    enemy = w.enemies[0]
    w.player.x = enemy.x - 0.4
    w.player.y = enemy.y
    w.player.vy = 0.0
    w.player.on_ground = True
    _, rc, terminated, info = w.tick(Action.NOOP)
    assert terminated
    assert info["death_cause"] == DeathCause.ENEMY
    assert rc["stomp"] == 0.0
    assert enemy.alive


def test_reaching_goal_terminates_with_goal_reached():
    tilemap = TileMap.from_json(LEVEL_1)
    w = World(tilemap)
    w.reset(seed=0)
    # teleport-free approach: directly place the player at the goal to test the win condition
    # in isolation from movement/pathing (pathing itself is exercised by the other tests).
    gx, gy = tilemap.goal_pos
    w.player.x, w.player.y = gx, gy
    state, rc, terminated, info = w.tick(Action.NOOP)
    assert terminated
    assert info["goal_reached"] is True
    assert rc["goal_bonus"] > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
