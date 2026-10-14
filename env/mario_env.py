"""Gymnasium wrapper around game.world.World.

Observation space (Box, float32, shape (OBS_DIM,)), in this exact order:
  [0]  vx, normalized by MAX_RUN_SPEED, clipped to [-1, 1]
  [1]  vy, normalized by TERMINAL_VELOCITY, clipped to [-1, 1]
  [2]  on_ground flag, {0, 1}
  [3]  facing_right flag, {-1, 1}
  [4:4+W*H]  local tile window (TILE_WINDOW_H x TILE_WINDOW_W), tile id / max_tile_id, flattened
             row-major, centered on the player's tile position
  [next 3*NUM_ENEMY_SLOTS] for each of the N nearest enemies (padded with the sentinel
             (1.0, 1.0, 0.0) -- "far away / idle" -- when fewer than N are alive):
             dx/ENEMY_RANGE clipped [-1,1], dy/ENEMY_RANGE clipped [-1,1],
             vx/ENEMY_SPEED clipped [-1,1]
  [-2] distance-to-goal, normalized by level width, in [0, 1] (0 == at the goal)
  [-1] time-remaining fraction, in [0, 1]

Un-normalized features are a classic reason PPO fails to learn, so every feature here is scaled
to roughly [-1, 1] or [0, 1]. See writeup.md for the rationale behind this exact vector.

Action space: Discrete(6), see game.constants.Action.

`terminated=True` on death (pit / enemy / hazard) or reaching the goal. `truncated=True` on the
max-episode-step timeout -- kept distinct so PPO bootstraps correctly on timeouts instead of
treating them as a true terminal state (see agent/train.py, which relies on SB3 reading these
two flags directly from the 5-tuple `step()` return, the modern Gymnasium/SB3 convention).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.constants import (
    ENEMY_H,
    ENEMY_SPEED,
    ENEMY_W,
    MAX_EPISODE_STEPS,
    MAX_RUN_SPEED,
    N_ACTIONS,
    PLAYER_H,
    PLAYER_W,
    TERMINAL_VELOCITY,
    Action,
    DeathCause,
)
from game.levels.level_gen import generate_level
from game.tilemap import TileMap
from game.world import World

TILE_WINDOW_W = 7
TILE_WINDOW_H = 5
MAX_TILE_ID = 5.0
NUM_ENEMY_SLOTS = 2
ENEMY_RANGE = 10.0  # tiles; relative enemy position/distance normalization range

OBS_DIM = 4 + TILE_WINDOW_W * TILE_WINDOW_H + 3 * NUM_ENEMY_SLOTS + 2


class MarioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        level_path: str = "game/levels/level_1.json",
        max_episode_steps: int = MAX_EPISODE_STEPS,
        level_paths: list[str] | None = None,
        level_weights: list[float] | None = None,
        procgen: bool = False,
        procgen_prob: float = 0.5,
    ):
        super().__init__()
        self.level_path = level_path
        self.level_paths = list(level_paths) if level_paths else []
        if level_weights is not None:
            w = np.asarray(level_weights, dtype=np.float64)
            assert len(w) == len(self.level_paths), "level_weights must match train_level_paths"
            self.level_weights = w / w.sum()
        else:
            self.level_weights = None
        self.procgen = procgen
        self.procgen_prob = float(procgen_prob)
        self.max_episode_steps = max_episode_steps

        self._set_tilemap(TileMap.from_json(level_path))
        self.world = World(self.tilemap)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    def _set_tilemap(self, tilemap: TileMap) -> None:
        self.tilemap = tilemap
        self._level_width = float(self.tilemap.width)

    def _sample_layout(self) -> None:
        """Pick a training layout for this episode (authored mix and/or procgen)."""
        use_procgen = self.procgen and (
            not self.level_paths or bool(self.np_random.random() < self.procgen_prob)
        )
        if use_procgen:
            seed = int(self.np_random.integers(0, 2**31 - 1))
            self._set_tilemap(TileMap.from_dict(generate_level(seed=seed)))
        elif self.level_paths:
            if self.level_weights is not None:
                idx = int(self.np_random.choice(len(self.level_paths), p=self.level_weights))
            else:
                idx = int(self.np_random.integers(0, len(self.level_paths)))
            self._set_tilemap(TileMap.from_json(self.level_paths[idx]))
        else:
            self._set_tilemap(TileMap.from_json(self.level_path))
        self.world = World(self.tilemap)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.level_paths or self.procgen:
            self._sample_layout()
        state = self.world.reset(seed=seed)
        obs = self._build_observation(state)
        info: dict[str, Any] = {
            "death_cause": int(DeathCause.NONE),
            "level_name": self.tilemap.name,
        }
        return obs, info

    def step(self, action: int):
        state, reward_components, terminated, info = self.world.tick(Action(action))
        reward = float(sum(reward_components.values()))

        truncated = False
        if not terminated and state.elapsed_steps >= self.max_episode_steps:
            truncated = True
            info["death_cause"] = int(DeathCause.TIMEOUT)

        obs = self._build_observation(state)
        info["reward_components"] = reward_components
        return obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError(
            "MarioEnv has no rendering of its own -- see render/renderer.py, which reads "
            "World state directly for the human-play / watch-agent renderer."
        )

    # ------------------------------------------------------------ observation
    def _build_observation(self, state) -> np.ndarray:
        player = state.player
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = np.clip(player.vx / MAX_RUN_SPEED, -1.0, 1.0)
        obs[1] = np.clip(player.vy / TERMINAL_VELOCITY, -1.0, 1.0)
        obs[2] = 1.0 if player.on_ground else 0.0
        obs[3] = 1.0 if player.facing_right else -1.0

        cx = player.x + PLAYER_W / 2
        cy = player.y + PLAYER_H / 2
        window = self.tilemap.local_window(cx, cy, TILE_WINDOW_W, TILE_WINDOW_H)
        flat = (window.astype(np.float32) / MAX_TILE_ID).flatten()
        obs[4:4 + flat.size] = flat

        offset = 4 + flat.size
        alive_enemies = [e for e in state.enemies if e.alive]
        dists = []
        for e in alive_enemies:
            ex, ey = e.x + ENEMY_W / 2, e.y + ENEMY_H / 2
            dx, dy = ex - cx, ey - cy
            dist = float(np.hypot(dx, dy))
            dists.append((dist, dx, dy, e.vx))
        dists.sort(key=lambda t: t[0])

        for i in range(NUM_ENEMY_SLOTS):
            base = offset + 3 * i
            if i < len(dists):
                _, dx, dy, evx = dists[i]
                obs[base] = np.clip(dx / ENEMY_RANGE, -1.0, 1.0)
                obs[base + 1] = np.clip(dy / ENEMY_RANGE, -1.0, 1.0)
                obs[base + 2] = np.clip(evx / ENEMY_SPEED, -1.0, 1.0)
            else:
                obs[base] = 1.0
                obs[base + 1] = 1.0
                obs[base + 2] = 0.0

        goal_x, _ = self.tilemap.goal_pos
        dist_to_goal = max(goal_x - cx, 0.0)
        obs[-2] = float(np.clip(dist_to_goal / self._level_width, 0.0, 1.0))
        obs[-1] = float(np.clip(1.0 - state.elapsed_steps / self.max_episode_steps, 0.0, 1.0))

        return obs
