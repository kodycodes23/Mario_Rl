"""Scripted random-action baseline: same env, uniform-random actions, same metrics as eval.

This is what "meaningfully beats random" is measured against in writeup.md -- run this and
`agent/play.py --headless` with the same `--episodes` and compare the printed tables directly.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from agent.common import build_env
from agent.callbacks import _DEATH_CAUSE_NAMES


def run_random_baseline(
    level_path: str,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> dict:
    env = build_env(
        level_path=level_path,
        max_episode_steps=max_episode_steps,
        frame_skip=1,
        sticky_action_p=0.0,
        reward_mode="shaped_time",
    )
    rng = np.random.default_rng(seed)

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
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
        if info.get("goal_reached"):
            completions += 1
        progresses.append(info.get("x_progress_frac", 0.0))
        survival_steps.append(steps)
        total_rewards.append(ep_reward)
        death_causes[_DEATH_CAUSE_NAMES.get(info.get("death_cause", 0), "unknown")] += 1

    results = {
        "episodes": episodes,
        "completion_rate": completions / episodes,
        "avg_progress_frac": float(np.mean(progresses)),
        "avg_survival_steps": float(np.mean(survival_steps)),
        "avg_reward": float(np.mean(total_rewards)),
        "death_cause_breakdown": {k: v / episodes for k, v in death_causes.items()},
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="game/levels/level_1.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="optional JSON output path")
    args = parser.parse_args()

    results = run_random_baseline(args.level, args.episodes, args.max_episode_steps, args.seed)
    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
