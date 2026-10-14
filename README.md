# Mario RL

A small, deterministic, headless-capable platformer (`game/`) wrapped as a Gymnasium env
(`env/`), trained with PPO via Stable-Baselines3 (`agent/`), with a separate, slick pygame-ce
renderer (`render/`) for human play and watching the trained agent.

**Stack:** Python 3.12 (spec called for 3.11; 3.12 was used here since 3.11 wasn't available
locally and every dependency below supports 3.12 identically), pygame-ce for rendering,
Gymnasium for the RL API, Stable-Baselines3 (PPO) for the RL algorithm. SB3 gives a *correct*
PPO loop for free, so the time budget goes into the game engine, the MDP formulation, and the
analysis in `writeup.md` -- that's where the grading signal is.

## Repo layout

See `mario_rl_build_prompt.md` for the full brief this was built from. Short version:

- `game/` -- pure Python + NumPy simulation core. No pygame import anywhere in this path, so it
  runs headless at thousands of steps/sec and stays trivially unit-testable.
- `env/` -- `MarioEnv(gymnasium.Env)` wrapping `game.world.World`, plus reward-shaping /
  frame-skip / sticky-action wrappers.
- `render/` -- pygame-ce renderer, menu, HUD, watch-agent policy overlay. Never imported by
  `env/`, `game/`, or `agent/train.py`.
- `agent/` -- `train.py` (PPO), `play.py` (checkpoint playback / headless eval),
  `random_baseline.py` (the comparison baseline).
- `configs/` -- YAML configs for the baseline and the reward-shaping ablation.
- `tests/` -- determinism + Gymnasium API compliance.
- `writeup.md` -- the research write-up: MDP formulation, reward design, results, stretch goals.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Level 1 only (Core agent):

```bash
python -m agent.train --config configs/ppo_baseline.yaml --seed 0
```

Curriculum up to level 4 (continue from the Core agent, **one new level per stage**, ~10 min each):

```bash
# Stage 1 — mix in level 2
python -m agent.train --config configs/ppo_curriculum_l2.yaml --seed 0 \
  --checkpoint runs/core_c_fix/best_model.zip --run-dir runs/curriculum_l2

python -m agent.play --checkpoint runs/curriculum_l2/best_model.zip --headless --episodes 50
python -m agent.play --checkpoint runs/curriculum_l2/best_model.zip \
  --level game/levels/level_2.json --headless --episodes 50

# Stage 2 — add level 3 (only if level 1 is still completing)
python -m agent.train --config configs/ppo_curriculum_l3.yaml --seed 0 \
  --checkpoint runs/curriculum_l2/best_model.zip --run-dir runs/curriculum_l3

# Stage 3 — add level 4
python -m agent.train --config configs/ppo_curriculum_l4.yaml --seed 0 \
  --checkpoint runs/curriculum_l3/best_model.zip --run-dir runs/curriculum_l4
```

Or all three stages in one go (~30–40 min): `bash scripts/train_up_to_l4.sh`

Do **not** use `configs/ppo_robust.yaml` from scratch — that mix forgot level 1.

Writes checkpoints, the best model, and TensorBoard logs to `runs/<timestamp>/`. Watch progress
with `tensorboard --logdir runs/<timestamp>/tb`.

## Evaluate / play back

```bash
# visual playback of a trained checkpoint
python -m agent.play --checkpoint runs/core_c_fix/best_model.zip --episodes 5 --render

# quantitative headless eval (completion rate, avg progress, death-cause breakdown)
python -m agent.play --checkpoint runs/core_c_fix/best_model.zip --headless --episodes 100

# random-action baseline, same metrics, for the writeup's comparison table
python -m agent.random_baseline --episodes 100
```

## Play it yourself / watch the agent in the UI

```bash
python -m render.app --checkpoint runs/<run>/best_model.zip
```

Opens a menu: Human play (arrow keys / WASD to move, space/up to jump), Watch agent (loads the
given checkpoint and renders its rollout with a live action-probability + value overlay), Level
select.

**Levels** (`game/levels/`), in increasing difficulty -- each verified beatable by a scripted
jump-over-obstacles bot (`tests/test_physics.py::test_all_levels_are_beatable`), and each built
with a validated minimum ~12-14 tile buffer between obstacles (tighter spacing was tried and
confirmed unfair -- a jump that clears one obstacle can carry straight into the next with no
room to land and re-jump):

| level | width (tiles) | enemies | pits | hazards | notes |
|---|---|---|---|---|---|
| `level_1` | 65 | 1 | 1 | 1 | Core training level |
| `level_2` | 90 | 2 | 2 | 1 | Held-out, for the generalization stretch goal |
| `level_3` | 126 | 2 | 2 | 2 | Medium -- longer, same obstacle density as level_1 |
| `level_4` | 162 | 3 | 3 | 2 | Hard -- more obstacles, longer sustained execution |
| `level_5` | 213 | 4 | 4 | 3 | Hardest -- most obstacles, longest level |

## Tests

```bash
python -m pytest tests/ -v
```

## Reproducibility

- `requirements.txt` is fully pinned (`pip freeze` after final setup).
- All randomness in `game/` goes through a single seeded `numpy.random.Generator` created in
  `World.reset(seed)` -- never the `random` module, never an unseeded call. See
  `tests/test_physics.py` for the determinism tests this guarantees.
- `agent/train.py` seeds `torch`, `numpy`, `random`, the env, and SB3's own `seed=` argument.
- **Hardware / wall-clock:** Apple M1 Pro, 8-core CPU only, ~27 min per 2M-step run
  (`configs/ppo_baseline.yaml`, `n_envs=8`, ~1200-1300 fps steady state). See `writeup.md` §4.

## Stretch goals attempted

1. **Reward-shaping ablation** (run: `configs/reward_sparse.yaml`, `configs/reward_shaped.yaml`,
   `configs/ppo_baseline.yaml`, 2M steps each, seed 0) -- sparse converges to a fully idle
   policy, `shaped_no_time` and `shaped_time` both learn real forward-progress-seeking behavior.
   Full results, plots, and a second local-optimum finding in `writeup.md` §5.1.
2. **Generalization** (run: `configs/ppo_generalization.yaml` + a corrected comparison using
   `configs/ppo_baseline.yaml`'s own checkpoint) -- surfaced a real model-selection methodology
   bug (test-set peeking) worth reading about; the corrected result is an honest
   negative/inconclusive finding. See `writeup.md` §5.2.
3. **Extended training** (5M steps vs. the standard 2M, same config/seed) -- shows the
   completion-rate plateau is a capability ceiling, not a training-budget problem. See
   `writeup.md` §3.2.
4. Observation Study (pixels + CNN) was **not attempted** -- see `writeup.md` for why.

## Disclosure

See the "AI assistance and disclosure" section in `writeup.md`.
