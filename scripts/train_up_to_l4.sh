#!/usr/bin/env bash
# Curriculum: continue from the Core level-1 agent, adding one harder map per stage.
# ~10 min per stage on M1 Pro CPU. After each stage, check that level 1 did not collapse
# before starting the next one.
set -euo pipefail
cd "$(dirname "$0")/.."

START="${1:-runs/core_c_fix/best_model.zip}"

echo "=== Stage 1: level 1 + 2  (~10 min) ==="
python -m agent.train --config configs/ppo_curriculum_l2.yaml --seed 0 \
  --checkpoint "$START" --run-dir runs/curriculum_l2
CKPT2=runs/curriculum_l2/best_model.zip
if [ ! -f "$CKPT2" ]; then CKPT2=runs/curriculum_l2/final_model.zip; fi
python -m agent.play --checkpoint "$CKPT2" --headless --episodes 50
python -m agent.play --checkpoint "$CKPT2" \
  --level game/levels/level_2.json --headless --episodes 50

echo "=== Stage 2: add level 3  (~10 min) ==="
python -m agent.train --config configs/ppo_curriculum_l3.yaml --seed 0 \
  --checkpoint "$CKPT2" --run-dir runs/curriculum_l3
CKPT3=runs/curriculum_l3/best_model.zip
if [ ! -f "$CKPT3" ]; then CKPT3=runs/curriculum_l3/final_model.zip; fi
python -m agent.play --checkpoint "$CKPT3" --headless --episodes 50
python -m agent.play --checkpoint "$CKPT3" \
  --level game/levels/level_3.json --headless --episodes 50

echo "=== Stage 3: add level 4  (~10 min) ==="
python -m agent.train --config configs/ppo_curriculum_l4.yaml --seed 0 \
  --checkpoint "$CKPT3" --run-dir runs/curriculum_l4
CKPT4=runs/curriculum_l4/best_model.zip
if [ ! -f "$CKPT4" ]; then CKPT4=runs/curriculum_l4/final_model.zip; fi
echo "Final evals:"
for L in 1 2 3 4; do
  echo "--- level $L ---"
  python -m agent.play --checkpoint "$CKPT4" \
    --level "game/levels/level_${L}.json" --headless --episodes 50
done
