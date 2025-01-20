import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os
import mujoco


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class QuadroboEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, xml_file="quad_robo.xml",
                 ctrl_cost_weight=0.15,
                 use_contact_forces=True,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 **kwargs):
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        xml_path = os.path.join(assets_dir, xml_file)
        utils.EzPickle.__init__(self, xml_path, 
                                ctrl_cost_weight,
                                use_contact_forces,
                                contact_cost_weight,
                                healthy_reward,
                                terminate_when_unhealthy,
                                healthy_z_range,
                                contact_force_range,
                                reset_noise_scale, 
                                **kwargs)
        
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self.obs_shape = 124

        self.init_qpos = np.zeros(self.obs_shape)  # Model's default positions
        self.init_qvel = np.zeros(self.obs_shape)  # Model's default velocities

        # Observation space includes positions and velocities, plus camera feeds
        observation_space = Box(
                low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float64
            )

        # Load the MuJoCo XML file
        MujocoEnv.__init__(self, xml_path, 5, observation_space=observation_space, **kwargs)


    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    
    def step(self, action):
        # Store xy position before simulation
        xy_position_before = self.get_body_com("root")[:2].copy()

        # Simulate the action
        self.do_simulation(action, self.frame_skip)

        # Store xy position after simulation
        xy_position_after = self.get_body_com("root")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Calculate forward and sideways displacements
        forward_displacement = xy_position_after[0] - xy_position_before[0]
        sideways_displacement = abs(xy_position_after[1] - xy_position_before[1])

        # Calculate default rewards
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        # velocity_error = abs(x_velocity - 0.5)

        # Gather observations
        observation = self._get_obs()
        # print(len(observation))
        # Parse observation
        joint_positions = observation[:12]
        joint_velocities = observation[12:24]
        roll, pitch, yaw = observation[121:124]

        total_reward = healthy_reward + 7.5*forward_displacement - 0.1*sideways_displacement + 5.0 * forward_reward - 0.5*np.sum(np.abs(joint_velocities)) 
        orientation_penalty = 0.02 * abs(roll) + 0.02*abs(pitch) + 0.05 *abs(yaw)

        # Total reward including penalties
        reward = total_reward - orientation_penalty

        # Define permissible ranges for roll, pitch, yaw
        max_roll = np.radians(60)  # Maximum roll in radians
        max_pitch = np.radians(60)  # Maximum pitch in radians
        max_yaw = np.radians(60)  # Maximum yaw in radians

        # Check if roll, pitch, or yaw exceed permissible ranges
        if abs(roll) > max_roll or abs(pitch) > max_pitch or abs(yaw) > max_yaw:
            terminated = True
        else:
            terminated = self.terminated

        # Debug information
        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "total_reward": total_reward,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self):


        # Get state observations (positions)
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")

        root_rot_matrix = self.data.xmat[root_body_id].reshape(3, 3)  # 3x3 rotation matrix

        # Convert rotation matrix to Euler angles
        roll = np.arctan2(root_rot_matrix[2, 1], root_rot_matrix[2, 2])
        pitch = np.arcsin(-root_rot_matrix[2, 0])
        yaw = np.arctan2(root_rot_matrix[1, 0], root_rot_matrix[0, 0])

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force, [roll, pitch, yaw]))
        else:
            return np.concatenate((position, velocity, [roll, pitch, yaw]))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
