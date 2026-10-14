"""Load a checkpoint and either watch it play (rendered) or run a quantitative headless eval.

Usage:
    python -m agent.play --checkpoint runs/<run>/best_model.zip --episodes 5 --render
    python -m agent.play --checkpoint runs/<run>/best_model.zip --headless --episodes 100
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from agent.callbacks import _DEATH_CAUSE_NAMES
from agent.common import build_env


def run_headless_eval(checkpoint: str, level_path: str, episodes: int,
                       max_episode_steps: int, seed: int, deterministic: bool = True) -> dict:
    model = PPO.load(checkpoint)
    env = build_env(
        level_path=level_path,
        max_episode_steps=max_episode_steps,
        frame_skip=1,
        sticky_action_p=0.0,
        reward_mode="shaped_time",
    )

    completions = 0
    progresses = []
    survival_steps = []
    total_rewards = []
    death_causes: Counter = Counter()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        steps = 0
        ep_reward = 0.0
        info: dict = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            steps += 1
        if info.get("goal_reached"):
            completions += 1
        progresses.append(info.get("x_progress_frac", 0.0))
        survival_steps.append(steps)
        total_rewards.append(ep_reward)
        death_causes[_DEATH_CAUSE_NAMES.get(info.get("death_cause", 0), "unknown")] += 1

    return {
        "episodes": episodes,
        "completion_rate": completions / episodes,
        "avg_progress_frac": float(np.mean(progresses)),
        "avg_survival_steps": float(np.mean(survival_steps)),
        "avg_reward": float(np.mean(total_rewards)),
        "death_cause_breakdown": {k: v / episodes for k, v in death_causes.items()},
    }


def run_rendered(checkpoint: str, level_path: str, episodes: int, max_episode_steps: int, seed: int):
    # renderer import kept local so headless eval / training never pulls in pygame
    from render.app import run_watch_agent

    model = PPO.load(checkpoint)
    run_watch_agent(model, level_path=level_path, episodes=episodes,
                     max_episode_steps=max_episode_steps, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--level", type=str, default="game/levels/level_1.json")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out", type=str, default=None, help="optional JSON output path (headless mode)")
    args = parser.parse_args()

    if args.render:
        run_rendered(args.checkpoint, args.level, args.episodes, args.max_episode_steps, args.seed)
    else:
        results = run_headless_eval(
            args.checkpoint, args.level, args.episodes, args.max_episode_steps, args.seed
        )
        print(json.dumps(results, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
