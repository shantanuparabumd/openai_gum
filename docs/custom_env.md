### **OpenAI Gym Environment: `QuadroboEnv`**

This section describes the `QuadroboEnv` custom environment built using OpenAI Gym and MuJoCo for simulating a quadruped robot. It includes reward design, termination conditions, and integration with the MuJoCo XML model.

---

### **Environment Configuration (`quadrobo.yaml`)**

The YAML file defines parameters to customize the environment:

| Parameter                  | Description                                                                                        | Default Value          |
|----------------------------|----------------------------------------------------------------------------------------------------|------------------------|
| `name`                     | Environment name.                                                                                 | `QuadroboEnv-v0`       |
| `forward_reward_weight`    | Weight for forward motion rewards.                                                                | `1.0`                  |
| `ctrl_cost_weight`         | Weight for penalizing control effort.                                                             | `0.05`                 |
| `contact_cost_weight`      | Weight for penalizing excessive contact forces.                                                   | `5e-4`                 |
| `healthy_reward`           | Reward for staying in a healthy state.                                                            | `1.0`                  |
| `terminate_when_unhealthy` | Whether the episode terminates when the robot becomes unhealthy.                                   | `true`                 |
| `healthy_z_range`          | Permissible height range of the torso.                                                            | `[0.2, 1.0]`           |
| `contact_force_range`      | Permissible range for contact forces.                                                             | `[-1.0, 1.0]`          |
| `render_mode`              | Rendering mode (`human`, `rgb_array`, `depth_array`).                                             | `human`                |
| `max_steps`                | Maximum steps per episode.                                                                        | `1000`                 |

---

### **Environment Code**

The `QuadroboEnv` class defines the core simulation logic, including:
- Observation space
- Reward computation
- Termination conditions

#### **Key Features**

1. **Observation Space (`_get_obs` method)**  
   The observation includes:
   - **Joint Positions**: First 12 elements (`qpos`).
   - **Joint Velocities**: Next 12 elements (`qvel`).
   - **Contact Forces**: If enabled, these values are clipped between `contact_force_range`.
   - **Root Orientation**: Extracted as roll, pitch, and yaw using a rotation matrix.

   ```python
   def _get_obs(self):
       position = self.data.qpos.flat.copy()
       velocity = self.data.qvel.flat.copy()
       root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")
       root_rot_matrix = self.data.xmat[root_body_id].reshape(3, 3)

       # Convert rotation matrix to Euler angles
       roll = np.arctan2(root_rot_matrix[2, 1], root_rot_matrix[2, 2])
       pitch = np.arcsin(-root_rot_matrix[2, 0])
       yaw = np.arctan2(root_rot_matrix[1, 0], root_rot_matrix[0, 0])

       if self._use_contact_forces:
           contact_force = self.contact_forces.flat.copy()
           return np.concatenate((position, velocity, contact_force, [roll, pitch, yaw]))
       else:
           return np.concatenate((position, velocity, [roll, pitch, yaw]))
   ```

2. **Reward Design (`step` method)**  
   - **Forward Reward**: Encourages forward motion by rewarding displacement along the X-axis.
   - **Sideways Penalty**: Penalizes displacement along the Y-axis.
   - **Joint Velocity Penalty**: Penalizes high joint velocities to encourage smooth motion.
   - **Orientation Penalty**: Penalizes roll, pitch, and yaw deviations beyond acceptable ranges.
   - **Control Cost**: Penalizes excessive control inputs.

   ```python
   forward_reward = x_velocity
   total_reward = (
       self.healthy_reward +
       7.5 * forward_displacement -
       0.1 * sideways_displacement +
       5.0 * forward_reward -
       0.5 * np.sum(np.abs(joint_velocities))
   )
   orientation_penalty = 0.02 * abs(roll) + 0.02 * abs(pitch) + 0.05 * abs(yaw)
   reward = total_reward - orientation_penalty
   ```

3. **Termination Conditions**
   - If roll, pitch, or yaw exceeds the defined limits (`±60°` in radians).
   - If the robot's height is outside the `healthy_z_range`.

   ```python
   if abs(roll) > max_roll or abs(pitch) > max_pitch or abs(yaw) > max_yaw:
       terminated = True
   else:
       terminated = self.terminated
   ```

4. **Action Space**  
   Controls are applied to 12 actuators (thigh, leg, and shin joints for each leg).

---

### **MuJoCo XML Model**

The XML file defines the robot's physical structure, joints, and actuators. Key components include:

1. **Model Parameters**
   - **Integrator**: `RK4` for stable simulations.
   - **Gravity**: Set to Earth's gravity (`-9.8 m/s²`).

   ```xml
   <option integrator="RK4" timestep="0.01" gravity="0 0 -9.8"/>
   ```

2. **Assets**  
   The robot's body and limbs are defined as meshes loaded from STL files.

   ```xml
   <mesh class="quadrobo" name="body" file="body.STL" />
   ```

3. **Body Definitions**  
   Each body part includes:
   - **Inertial Properties**: Mass and inertia.
   - **Joint Definitions**: Range of motion and axis of rotation.
   - **Collision and Visual Geometries**: Defined separately for collision detection and rendering.

   ```xml
   <body name="root" pos="0 0 1.0" quat="1 0 0 0">
       <freejoint name="root" />
       <geom type="mesh" rgba="0 0.752941 0 1" mesh="body" class="visual" />
   </body>
   ```

4. **Actuators**  
   Position actuators control the joints.

   ```xml
   <actuator>
       <position class="quadrobo" name="back_right_thigh_joint" joint="back_right_thigh_joint" />
   </actuator>
   ```

---

### **Usage**
1. **Register the Environment**
   ```python
   from gym.envs.registration import register
   register(
       id="QuadroboEnv-v0",
       entry_point="environments.quad_env:QuadroboEnv",
   )
   ```

2. **Running the Environment**
   ```python
   import gym
   env = gym.make("QuadroboEnv-v0", render_mode="human")
   state = env.reset()
   for _ in range(1000):
       action = env.action_space.sample()
       state, reward, done, _, info = env.step(action)
       if done:
           break
   env.close()
   ```

---

This design provides a robust simulation environment for training reinforcement learning policies on a quadruped robot. The modular reward design and customizable parameters make it versatile for various tasks and objectives.