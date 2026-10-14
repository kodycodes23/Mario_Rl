from gymnasium.envs.registration import register

register(
    id="MarioRL-v0",
    entry_point="env.mario_env:MarioEnv",
    max_episode_steps=None,  # MarioEnv handles its own truncation; see constants.MAX_EPISODE_STEPS
)
