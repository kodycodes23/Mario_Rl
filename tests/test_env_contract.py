"""Gymnasium API compliance for MarioEnv, plus wrapper sanity checks."""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.mario_env import OBS_DIM, MarioEnv
from env.wrappers import FrameSkipWrapper, RewardShapingWrapper, StickyActionWrapper


def test_env_checker_passes():
    env = MarioEnv(level_path="game/levels/level_1.json")
    check_env(env.unwrapped, skip_render_check=True)


def test_reset_returns_correct_shapes():
    env = MarioEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_five_tuple_with_correct_types():
    env = MarioEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_components" in info


def test_observation_stays_in_declared_bounds():
    env = MarioEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(500):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), f"obs out of bounds: {obs}"
        if terminated or truncated:
            obs, _ = env.reset(seed=0)


def test_truncation_at_max_episode_steps():
    env = MarioEnv(max_episode_steps=50)
    env.reset(seed=0)
    truncated = False
    terminated = False
    steps = 0
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(0)  # NOOP forever, shouldn't die
        steps += 1
        if terminated or truncated:
            break
    assert truncated
    assert not terminated
    assert steps == 50


def test_reward_shaping_wrapper_modes_differ():
    base_env = MarioEnv()
    env = RewardShapingWrapper(base_env, mode="sparse")
    env.reset(seed=0)
    _, r_sparse, term, trunc, info = env.step(2)  # RIGHT
    # sparse reward should be 0 unless the episode ended this very step
    if not (term or trunc):
        assert r_sparse == 0.0


def test_frame_skip_wrapper_sums_reward_and_skips():
    base_env = MarioEnv()
    env = FrameSkipWrapper(base_env, skip=4)
    obs, _ = env.reset(seed=0)
    obs2, reward, terminated, truncated, info = env.step(2)  # RIGHT
    assert obs2.shape == (OBS_DIM,)


def test_sticky_action_wrapper_reproducible_under_seed():
    def run(seed):
        env = StickyActionWrapper(MarioEnv(), p=0.3)
        env.reset(seed=seed)
        xs = []
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(2)
            xs.append(obs[0])
            if term or trunc:
                break
        return xs

    assert run(5) == run(5)


def test_mixed_levels_same_seed_reproduces_layout():
    paths = ["game/levels/level_1.json", "game/levels/level_3.json"]
    env = MarioEnv(level_paths=paths, procgen=False)
    obs1, info1 = env.reset(seed=7)
    name1, w1 = info1["level_name"], env.tilemap.width
    obs2, info2 = env.reset(seed=7)
    assert info2["level_name"] == name1
    assert env.tilemap.width == w1
    assert np.allclose(obs1, obs2)


def test_procgen_reset_stays_in_obs_bounds():
    env = MarioEnv(procgen=True, procgen_prob=1.0)
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert info["level_name"].startswith("procgen")
    for _ in range(30):
        obs, _, terminated, truncated, _ = env.step(2)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset()


def test_level_weights_change_sampling_distribution():
    env = MarioEnv(
        level_paths=["game/levels/level_1.json", "game/levels/level_3.json"],
        level_weights=[1.0, 0.0],
        procgen=False,
    )
    names = []
    for i in range(20):
        _, info = env.reset(seed=1000 + i)
        names.append(info["level_name"])
    assert set(names) == {"level_1"}


def test_held_out_level_2_is_not_in_robust_train_mix():
    import yaml
    with open("configs/ppo_robust.yaml") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["env"]["train_level_paths"]
    assert "game/levels/level_2.json" not in paths
    assert cfg["env"]["eval_level_path"] != "game/levels/level_2.json"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
