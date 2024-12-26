from gymnasium.envs.registration import register

register(
    id="QuadRobot-v0",
    entry_point="environments.quad_env:QuadRobotEnv",
    max_episode_steps=1000,
)
