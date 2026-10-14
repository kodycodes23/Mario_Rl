# Hand-rolled PPO (not attempted)

This directory is a placeholder. Per `mario_rl_build_prompt.md` §6, a from-scratch PPO
implementation is optional and lowest-ROI relative to the reward-shaping ablation and the
generalization stretch goal, both of which we did attempt (see `writeup.md`). Re-deriving PPO's
clipped surrogate objective, GAE, and minibatch update loop by hand would mostly re-prove that
Stable-Baselines3's implementation is correct, at the cost of time better spent on the MDP
formulation, level design, and analysis -- which is where the grading signal actually is per the
challenge brief. See `writeup.md`'s "Stretch goals attempted" section for the explicit
trade-off writeup.
