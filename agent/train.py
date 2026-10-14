"""Train PPO from scratch, or continue from a saved checkpoint.

Usage:
    python -m agent.train --config configs/ppo_baseline.yaml --seed 0
    python -m agent.train --config configs/ppo_curriculum_l2.yaml \\
        --checkpoint runs/core_c_fix/best_model.zip --run-dir runs/curriculum_l2
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent.callbacks import CompletionRateCallback, MultiLevelEvalCallback
from agent.common import load_config, make_env_fn, set_global_seeds


def train(
    config_path: str,
    seed_override: int | None,
    run_dir: Path | None = None,
    checkpoint: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = seed_override
    seed = cfg["seed"]
    set_global_seeds(seed)

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy(config_path, run_dir / "config.yaml")
    with open(run_dir / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    env_cfg = cfg["env"]
    n_envs = env_cfg["n_envs"]
    env_fns = [make_env_fn(cfg, seed, rank=i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else env_fns[0]()

    ppo_cfg = dict(cfg["ppo"])
    total_timesteps = ppo_cfg.pop("total_timesteps")
    policy = ppo_cfg.pop("policy")

    if checkpoint:
        print(f"Continuing from {checkpoint}")
        model = PPO.load(
            checkpoint,
            env=vec_env,
            seed=seed,
            tensorboard_log=str(run_dir / "tb"),
            verbose=1,
        )
        lr = cfg["ppo"].get("learning_rate")
        if lr is not None:
            model.learning_rate = lr
    else:
        model = PPO(
            policy,
            vec_env,
            seed=seed,
            tensorboard_log=str(run_dir / "tb"),
            verbose=1,
            **ppo_cfg,
        )

    eval_cfg = cfg["eval"]
    checkpoint_cfg = cfg["checkpoint"]
    eval_levels = eval_cfg.get("level_paths") or [env_cfg["eval_level_path"]]

    cb_list = [
        CheckpointCallback(
            save_freq=max(checkpoint_cfg["save_freq"] // n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_mario",
        ),
        CompletionRateCallback(
            eval_env=make_env_fn(cfg, seed + 20_000, level_path=env_cfg["eval_level_path"])(),
            eval_freq=eval_cfg["eval_freq"],
            n_eval_episodes=eval_cfg["n_eval_episodes"],
            deterministic=eval_cfg["deterministic"],
            verbose=1,
        ),
    ]

    if len(eval_levels) > 1:
        eval_envs = {}
        for i, path in enumerate(eval_levels):
            name = Path(path).stem
            eval_envs[name] = make_env_fn(cfg, seed + 30_000 + i, level_path=path)()
        cb_list.insert(0, MultiLevelEvalCallback(
            eval_envs=eval_envs,
            save_path=str(run_dir),
            eval_freq=eval_cfg["eval_freq"],
            n_eval_episodes=min(eval_cfg["n_eval_episodes"], 10),
            min_l1_completion=float(eval_cfg.get("min_l1_completion", 0.5)),
            deterministic=eval_cfg["deterministic"],
            verbose=1,
        ))
    else:
        eval_env = make_env_fn(cfg, seed + 10_000, level_path=env_cfg["eval_level_path"])()
        cb_list.insert(0, EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=max(eval_cfg["eval_freq"] // n_envs, 1),
            n_eval_episodes=eval_cfg["n_eval_episodes"],
            deterministic=eval_cfg["deterministic"],
        ))

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(cb_list),
        progress_bar=False,
        reset_num_timesteps=checkpoint is None,
    )
    model.save(str(run_dir / "final_model"))

    if isinstance(vec_env, SubprocVecEnv):
        vec_env.close()

    print(f"Training complete. Artifacts in {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_baseline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Existing PPO zip to continue from (e.g. runs/core_c_fix/best_model.zip)",
    )
    args = parser.parse_args()

    train(
        args.config,
        args.seed,
        Path(args.run_dir) if args.run_dir else None,
        checkpoint=args.checkpoint,
    )
