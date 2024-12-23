import gymnasium as gym
from typing import Tuple
from utils.rewards import calculate_reward


class AntEnv:
    def __init__(self, env_config, reward_config):
        """Initialize the Ant environment with custom configurations.

        Args:
            env_config: Configuration dictionary for the environment.
            reward_config: Configuration dictionary for custom rewards.
        """
        self.env = gym.make(
            env_config.name,
            render_mode=env_config.render_mode,
            forward_reward_weight=env_config.forward_reward_weight,
            ctrl_cost_weight=env_config.ctrl_cost_weight,
            contact_cost_weight=env_config.contact_cost_weight,
            healthy_reward=env_config.healthy_reward,
            terminate_when_unhealthy=env_config.terminate_when_unhealthy,
            healthy_z_range=env_config.healthy_z_range,
            contact_force_range=env_config.contact_force_range,
            include_cfrc_ext_in_observation=True,
        )
        self.max_steps = env_config.max_steps
        self.reward_config = reward_config

    def reset(self) -> Tuple:
        """Reset the environment and return the initial state."""
        return self.env.reset()

    def step(self, action) -> Tuple:
        """Take a step in the environment and calculate the reward.

        Args:
            action: The action to apply in the environment.

        Returns:
            Tuple containing the next state, custom reward, termination status,
            truncation status, and additional info.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Calculate custom reward
        custom_reward = calculate_reward(next_state, reward, self.reward_config)

        return next_state, custom_reward, terminated, truncated, info

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    @property
    def observation_space(self):
        """Return the observation space of the environment."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Return the action space of the environment."""
        return self.env.action_space

    def render(self) -> None:
        """Render the environment using the pre-set render mode."""
        self.env.render()
