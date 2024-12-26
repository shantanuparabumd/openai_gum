import os
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from gymnasium.utils import EzPickle
import mujoco

class QuadRobotEnv(gym.Env, EzPickle):
    """
    Custom Gymnasium environment for the QuadRobot using MuJoCo.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, xml_file="quad_robo.xml", render_mode=None):
        EzPickle.__init__(self, xml_file, render_mode)

        # Path to the MJCF file
        xml_path = os.path.join(os.path.dirname(__file__), "../assets", xml_file)

        import mujoco  # Import Mujoco only when needed to keep dependencies light
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Set observation and action spaces
        obs_dim = self.model.nq + self.model.nv  # positions (qpos) + velocities (qvel)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Render mode setup
        self.render_mode = render_mode
        self.viewer = None  # Viewer will be initialized lazily during render()

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Observation after reset
        observation = self.normalize_observation(np.concatenate([self.data.qpos, self.data.qvel]))

        return observation, {}

    def step(self, action):
        mujoco.mj_step(self.model, self.data)

        # Observation after step
        observation = self.normalize_observation(np.concatenate([self.data.qpos, self.data.qvel]))

        # Calculate reward
        reward = self._calculate_reward()

        # Define termination criteria
        terminated = self._is_terminated()
        truncated = False  # Add truncation logic if needed

        # Return step data
        return observation, reward, terminated, truncated, {}

    def normalize_observation(self, observation):
        # Normalize qpos and qvel separately
        normalized_qpos = observation[:self.model.nq] / np.clip(self.model.nq, a_min=1.0, a_max=None)
        normalized_qvel = observation[self.model.nq:] / np.clip(self.model.nv, a_min=1.0, a_max=None)
        return np.concatenate([normalized_qpos, normalized_qvel]).astype(np.float32)


    def _calculate_reward(self):
        # Example reward: forward velocity and penalties for sideways velocity
        forward_velocity = self.data.qvel[0]
        sideways_velocity = abs(self.data.qvel[1])
        return forward_velocity - 0.1 * sideways_velocity

    def _is_terminated(self):
        # Example termination: based on height of the torso
        torso_height = self.data.qpos[2]
        return torso_height < 0.2 or torso_height > 1.0

    def render(self):
        """Handles rendering based on the mode."""
        if self.render_mode == "human":
            if self.viewer is None:
                # Initialize the viewer only if it hasn't been initialized
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.model, self.data)
            # No explicit render call needed; launch_passive handles the GUI.
        elif self.render_mode == "rgb_array":
            from mujoco.renderer import MjRenderer
            # Use an offscreen renderer for pixel data
            renderer = MjRenderer(self.model)
            renderer.render(self.data)
            return renderer.read_pixels()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")


    def close(self):
        """Properly closes the viewer."""
        if self.viewer is not None:
            if hasattr(self.viewer, "close"):
                self.viewer.close()
                self.viewer = None

