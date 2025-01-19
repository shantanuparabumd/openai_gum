from gym.envs.registration import register

register(
    id="QuadroboEnv-v0",  # Environment ID
    entry_point="quadrobo_gym.env:QuadroboEnv",  # Ensure this matches your module structure
)
