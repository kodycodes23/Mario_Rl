"""Shared config loading, env-factory, seeding, and eval helpers for train/play/random_baseline.

Kept out of `env/` on purpose -- this module is allowed to import SB3/torch, `env/` and `game/`
must never import either, so that training speed is never affected by anything import-adjacent
to the RL stack.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from env.mario_env import MarioEnv
from env.wrappers import FrameSkipWrapper, RewardShapingWrapper, StickyActionWrapper


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "env": {
        "level_path": "game/levels/level_1.json",
        "eval_level_path": "game/levels/level_1.json",
        "train_level_paths": None,
        "train_level_weights": None,
        "procgen": False,
        "procgen_prob": 0.5,
        "max_episode_steps": 2400,
        "frame_skip": 2,
        "sticky_action_p": 0.0,
        "reward_mode": "shaped_time",
        "n_envs": 8,
    },
    "ppo": {
        "policy": "MlpPolicy",
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "learning_rate": 3e-4,
        "total_timesteps": 2_000_000,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    "eval": {"eval_freq": 20000, "n_eval_episodes": 20, "deterministic": True},
    "checkpoint": {"save_freq": 100000},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    return cfg


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def build_env(
    level_path: str,
    max_episode_steps: int,
    frame_skip: int,
    sticky_action_p: float,
    reward_mode: str,
    level_paths: list[str] | None = None,
    level_weights: list[float] | None = None,
    procgen: bool = False,
    procgen_prob: float = 0.5,
):
    """Returns a fully-wrapped, un-vectorized MarioEnv, ready for Monitor wrapping by the caller."""
    env = MarioEnv(
        level_path=level_path,
        max_episode_steps=max_episode_steps,
        level_paths=level_paths,
        level_weights=level_weights,
        procgen=procgen,
        procgen_prob=procgen_prob,
    )
    env = RewardShapingWrapper(env, mode=reward_mode)
    if sticky_action_p > 0:
        env = StickyActionWrapper(env, p=sticky_action_p)
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)
    return env


def make_env_fn(cfg: dict[str, Any], seed: int, level_path: str | None = None, rank: int = 0):
    """Returns a zero-arg factory suitable for SB3's `make_vec_env` / `SubprocVecEnv`."""
    env_cfg = cfg["env"]
    resolved_level_path = level_path or env_cfg["level_path"]
    # Eval passes an explicit level_path; training mix/procgen only applies to train envs.
    mixing = level_path is None
    train_paths = env_cfg.get("train_level_paths") if mixing else None
    train_weights = env_cfg.get("train_level_weights") if mixing else None
    procgen = bool(env_cfg.get("procgen", False)) if mixing else False
    procgen_prob = float(env_cfg.get("procgen_prob", 0.5))

    def _init():
        from stable_baselines3.common.monitor import Monitor

        env = build_env(
            level_path=resolved_level_path,
            max_episode_steps=env_cfg["max_episode_steps"],
            frame_skip=env_cfg["frame_skip"],
            sticky_action_p=env_cfg["sticky_action_p"],
            reward_mode=env_cfg["reward_mode"],
            level_paths=train_paths,
            level_weights=train_weights,
            procgen=procgen,
            procgen_prob=procgen_prob,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
