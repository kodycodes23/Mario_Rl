"""Env wrappers: frame-skip, reward-shaping variants, sticky actions, obs normalization.

`MarioEnv` itself always returns the raw shaped-baseline reward (sum of all reward components)
plus the per-component breakdown in `info["reward_components"]`. `RewardShapingWrapper`
recomputes the scalar reward from that breakdown according to `mode`, which is how the
sparse-vs-shaped ablation (see configs/reward_sparse.yaml, configs/reward_shaped.yaml) is
implemented without touching game/world.py at all.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

REWARD_MODES = ("sparse", "shaped_no_time", "shaped_time")


class RewardShapingWrapper(gym.Wrapper):
    """Recomputes the scalar reward from info["reward_components"] for the ablation study.

    - "sparse":        goal_bonus + death_penalty only (no progress shaping, no time penalty)
    - "shaped_no_time": progress + position + stomp + goal_bonus + death_penalty
    - "shaped_time":    those plus the per-step time penalty (the Core default)
    """

    def __init__(self, env: gym.Env, mode: str = "shaped_time"):
        super().__init__(env)
        assert mode in REWARD_MODES, f"unknown reward mode {mode!r}, expected one of {REWARD_MODES}"
        self.mode = mode

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        rc = info["reward_components"]
        if self.mode == "sparse":
            reward = rc["goal_bonus"] + rc["death_penalty"]
        elif self.mode == "shaped_no_time":
            reward = (
                rc["progress"] + rc["position"] + rc["stomp"]
                + rc["goal_bonus"] + rc["death_penalty"]
            )
        else:
            reward = (
                rc["progress"] + rc["position"] + rc["stomp"] + rc["time_penalty"]
                + rc["goal_bonus"] + rc["death_penalty"]
            )
        return obs, float(reward), terminated, truncated, info


class FrameSkipWrapper(gym.Wrapper):
    """Repeats the chosen action for `skip` sim ticks, summing reward and returning the last obs.

    Ends early (without stepping further) if the episode terminates or truncates mid-skip.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        assert skip >= 1
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, info, terminated, truncated = None, {}, False, False
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class StickyActionWrapper(gym.Wrapper):
    """With probability `p`, repeats the previous action instead of the newly chosen one.

    Adds mild stochasticity (classic Atari-style "sticky actions") so the policy can't rely on
    perfectly deterministic dynamics -- driven entirely by the env's own seeded RNG so episodes
    stay reproducible under a fixed seed.
    """

    def __init__(self, env: gym.Env, p: float = 0.1):
        super().__init__(env)
        assert 0.0 <= p < 1.0
        self.p = p
        self._prev_action = 0
        self._sticky_rng: np.random.Generator = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._sticky_rng = np.random.default_rng(seed)
        self._prev_action = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self._sticky_rng.random() < self.p:
            action = self._prev_action
        else:
            self._prev_action = action
        return self.env.step(action)


def make_reward_wrapper(env: gym.Env, mode: str) -> gym.Env:
    return RewardShapingWrapper(env, mode=mode)
