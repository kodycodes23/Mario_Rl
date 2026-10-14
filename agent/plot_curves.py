"""Render learning-curve PNGs from TensorBoard logs for writeup.md.

Not part of the training/eval path -- a one-off analysis script, run manually:
    python -m agent.plot_curves
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "run 1 (death=-50, time=-0.01)": "runs/20260919-175313_local_optimum_death50",
    "run 2 (death=-15, time=-0.02, 2M steps)": "runs/20260919-182317",
    "run 3 (same as run 2, 5M steps)": "runs/20260930-083836",
}
COLORS = {
    "run 1 (death=-50, time=-0.01)": "#d9534f",
    "run 2 (death=-15, time=-0.02, 2M steps)": "#337ab7",
    "run 3 (same as run 2, 5M steps)": "#5cb85c",
}

ABLATION_RUNS = {
    "sparse (goal+death only)": "runs/ablation_sparse",
    "shaped_no_time (+progress)": "runs/ablation_shaped_no_time",
    "shaped_time (+time penalty)": "runs/20260919-182317",
}
ABLATION_COLORS = {
    "sparse (goal+death only)": "#d9534f",
    "shaped_no_time (+progress)": "#f0ad4e",
    "shaped_time (+time penalty)": "#337ab7",
}

CORE_C_RUNS = {
    "PPO Core C (stomp + live-x, 2M)": "runs/core_c_fix",
}
CORE_C_COLORS = {
    "PPO Core C (stomp + live-x, 2M)": "#337ab7",
}


def load_scalar(run_dir: str, tag: str):
    tb_dirs = sorted(Path(run_dir).glob("tb/PPO_*"))
    assert tb_dirs, f"no tb logs under {run_dir}"
    ea = EventAccumulator(str(tb_dirs[0]))
    ea.Reload()
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def plot(tag: str, ylabel: str, title: str, out_path: str, smooth: int = 1,
         runs: dict | None = None, colors: dict | None = None):
    runs = runs if runs is not None else RUNS
    colors = colors if colors is not None else COLORS
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, run_dir in runs.items():
        steps, values = load_scalar(run_dir, tag)
        if smooth > 1 and len(values) > smooth:
            import numpy as np
            kernel = np.ones(smooth) / smooth
            values = list(np.convolve(values, kernel, mode="valid"))
            steps = steps[smooth - 1:]
        ax.plot(steps, values, label=label, color=colors[label], linewidth=1.5)
    ax.set_xlabel("training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="writeup_assets")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    plot("rollout/ep_rew_mean", "mean episode reward", "Training reward vs. steps",
         f"{args.out_dir}/reward_curve.png", smooth=5)
    plot("eval/completion_rate", "completion rate", "Eval completion rate vs. steps",
         f"{args.out_dir}/completion_rate_curve.png")
    plot("eval/avg_progress_frac", "avg progress toward goal", "Eval progress fraction vs. steps",
         f"{args.out_dir}/progress_curve.png")
    plot("rollout/ep_len_mean", "mean episode length (wrapped steps)", "Episode length vs. steps",
         f"{args.out_dir}/ep_len_curve.png", smooth=5)

    plot("eval/completion_rate", "completion rate", "Ablation: eval completion rate vs. steps",
         f"{args.out_dir}/ablation_completion_rate.png",
         runs=ABLATION_RUNS, colors=ABLATION_COLORS)
    plot("eval/avg_progress_frac", "avg progress toward goal", "Ablation: eval progress fraction vs. steps",
         f"{args.out_dir}/ablation_progress.png",
         runs=ABLATION_RUNS, colors=ABLATION_COLORS)
    plot("rollout/ep_rew_mean", "mean training-time reward (own reward scale)",
         "Ablation: training reward vs. steps (not cross-comparable, see caption)",
         f"{args.out_dir}/ablation_reward.png", smooth=5,
         runs=ABLATION_RUNS, colors=ABLATION_COLORS)

    plot("rollout/ep_rew_mean", "mean episode reward", "Core C: training reward vs. steps",
         f"{args.out_dir}/core_c_reward_curve.png", smooth=5,
         runs=CORE_C_RUNS, colors=CORE_C_COLORS)
    plot("eval/completion_rate", "completion rate", "Core C: eval completion rate vs. steps",
         f"{args.out_dir}/core_c_completion_rate_curve.png",
         runs=CORE_C_RUNS, colors=CORE_C_COLORS)
    plot("eval/avg_progress_frac", "avg progress toward goal", "Core C: eval progress vs. steps",
         f"{args.out_dir}/core_c_progress_curve.png",
         runs=CORE_C_RUNS, colors=CORE_C_COLORS)
