"""Custom SB3 callback: completion rate + death-cause breakdown, logged to TensorBoard.

SB3's built-in EvalCallback logs mean reward / episode length, which is necessary but not
sufficient -- reward curves alone can be misleading (see mario_rl_build_prompt.md §4). This
callback runs its own eval loop against a single (non-vectorized) eval env and logs the metric
that actually answers "did it learn to play": level completion rate.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from game.constants import DeathCause

_DEATH_CAUSE_NAMES = {
    int(DeathCause.NONE): "none",
    int(DeathCause.PIT): "pit",
    int(DeathCause.ENEMY): "enemy",
    int(DeathCause.HAZARD): "hazard",
    int(DeathCause.TIMEOUT): "timeout",
    int(DeathCause.GOAL): "goal",
}


class CompletionRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 20,
                 deterministic: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        self._run_eval()
        return True

    def _run_eval(self) -> None:
        completions = 0
        progresses = []
        survival_steps = []
        death_causes = Counter()

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            terminated = truncated = False
            steps = 0
            info: dict = {}
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _reward, terminated, truncated, info = self.eval_env.step(int(action))
                steps += 1
            if info.get("goal_reached"):
                completions += 1
            progresses.append(info.get("x_progress_frac", 0.0))
            survival_steps.append(steps)
            death_causes[_DEATH_CAUSE_NAMES.get(info.get("death_cause", 0), "unknown")] += 1

        completion_rate = completions / self.n_eval_episodes
        self.logger.record("eval/completion_rate", completion_rate)
        self.logger.record("eval/avg_progress_frac", float(np.mean(progresses)))
        self.logger.record("eval/avg_survival_steps", float(np.mean(survival_steps)))
        for cause, count in death_causes.items():
            self.logger.record(f"eval/death_cause_{cause}", count / self.n_eval_episodes)
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"[CompletionRateCallback] step={self.num_timesteps} "
                f"completion_rate={completion_rate:.2f} "
                f"avg_progress={float(np.mean(progresses)):.2f}"
            )


class MultiLevelEvalCallback(BaseCallback):
    """Eval several fixed levels. Save best only if level 1 stays competent.

    Score = 0.4 * L1 completion + 0.6 * mean completion across all listed levels.
    Refuses to save if level-1 completion drops below `min_l1_completion`.
    """

    def __init__(
        self,
        eval_envs: dict[str, object],
        save_path: str,
        eval_freq: int,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        min_l1_completion: float = 0.5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_envs = eval_envs
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.min_l1_completion = min_l1_completion
        self._last_eval_step = 0
        self.best_score = -1.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        rates = {}
        for name, env in self.eval_envs.items():
            completions = 0
            for _ in range(self.n_eval_episodes):
                obs, _ = env.reset()
                terminated = truncated = False
                info: dict = {}
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, _, terminated, truncated, info = env.step(int(action))
                if info.get("goal_reached"):
                    completions += 1
            rates[name] = completions / self.n_eval_episodes
            self.logger.record(f"eval/{name}_completion", rates[name])

        l1 = rates.get("level_1", 1.0)
        mean_all = float(np.mean(list(rates.values())))
        score = 0.4 * l1 + 0.6 * mean_all
        self.logger.record("eval/curriculum_score", score)
        self.logger.dump(self.num_timesteps)

        saved = False
        if l1 >= self.min_l1_completion and score > self.best_score:
            self.best_score = score
            self.model.save(f"{self.save_path}/best_model")
            saved = True
        if self.verbose:
            bits = " ".join(f"{k}={v:.2f}" for k, v in rates.items())
            print(
                f"[MultiLevelEval] step={self.num_timesteps} {bits} "
                f"score={score:.2f}{' *saved*' if saved else ''}"
            )
        return True
