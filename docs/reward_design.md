# Reward Design Explanation for the Quadrobo Environment

The reward system in the Quadrobo environment encourages efficient locomotion while maintaining stability and penalizing undesired behaviors. The design integrates multiple reward components and penalties to shape the desired behavior of the robot.

---

## Components of the Reward Function

### 1. **Forward Displacement Reward**
- **Purpose:** Incentivizes the robot to move forward along the x-axis.
- **Calculation:**
  $
  \text{Forward Displacement Reward} = 7.5 \times \text{forward\_displacement}
  $
  Where:
  - `forward_displacement` is the change in the robot's position along the x-axis.
  - The multiplier \(7.5\) weights this component to prioritize forward movement.

---

### 2. **Sideways Displacement Penalty**
- **Purpose:** Penalizes deviations along the y-axis to encourage the robot to maintain a straight path.
- **Calculation:**
  $
  \text{Sideways Penalty} = -0.1 \times \text{sideways\_displacement}
  $
  Where:
  - `sideways_displacement` is the absolute change in position along the y-axis.
  - The weight \(0.1\) scales the penalty.

---

### 3. **Velocity Reward**
- **Purpose:** Rewards the robot for maintaining a higher forward velocity.
- **Calculation:**
  $
  \text{Velocity Reward} = 5.0 \times \text{forward\_velocity}
  $
  Where:
  - `forward_velocity` is the robot's velocity along the x-axis.
  - The weight \(5.0\) emphasizes the importance of maintaining speed.

---

### 4. **Joint Velocity Penalty**
- **Purpose:** Penalizes high joint velocities to encourage smoother movements and energy efficiency.
- **Calculation:**
  $
  \text{Joint Velocity Penalty} = -0.5 \times \sum |\text{joint\_velocities}|
  $
  Where:
  - `joint_velocities` represents the velocities of the robot's joints.

---

### 5. **Orientation Penalty**
- **Purpose:** Penalizes deviations in the robot's orientation (roll, pitch, and yaw).
- **Calculation:**
  $
  \text{Orientation Penalty} = - \left( 0.02 \times |\text{roll}| + 0.02 \times |\text{pitch}| + 0.05 \times |\text{yaw}| \right)
  $
  Where:
  - `roll`, `pitch`, and `yaw` are the robot's orientation angles in radians.
  - The weights (\(0.02, 0.02, 0.05\)) scale the penalties for each angle, with yaw deviations penalized more heavily.

---

### 6. **Survival Reward**
- **Purpose:** Rewards the robot for staying operational and avoiding termination conditions.
- **Calculation:**
  $
  \text{Survival Reward} = \text{healthy\_reward}
  $
  
  Where:
  - `healthy_reward` is a fixed value (e.g., (1.0)) awarded at each timestep to encourage survival.$
---

## Termination Conditions

The episode terminates early if any of the following conditions are met:
1. **Orientation Limits:** 
   - Roll, pitch, or yaw exceeds $( \pm 60^\circ ) (( \pm 1.047 ) radians)$.
   - These limits ensure the robot avoids unstable orientations.

2. **Height Constraint:**
   - The robot's torso height falls outside the defined `healthy_z_range`.

3. **Environment-Specific Termination:**
   - Custom conditions defined in the environment.

---

## Total Reward Calculation

The total reward combines the components:
$
\text{Total Reward} = \text{Survival Reward} + \text{Forward Displacement Reward} + \text{Velocity Reward} - \left( \text{Sideways Penalty} + \text{Joint Velocity Penalty} + \text{Orientation Penalty} \right)
$

---

## Debug Information

To assist with monitoring and debugging, the following metrics are logged for each timestep:
- **Forward Reward:** Contribution of forward motion.
- **Survival Reward:** Indicates whether the robot remained operational.
- **Position:** \(x, y\)-coordinates of the robot.
- **Velocity:** \(x, y\)-velocity components.
- **Orientation:** Roll, pitch, and yaw values.
- **Total Reward:** Sum of all reward components.

---

## Example Weights and Parameters

Below is an example configuration for the reward components:

| **Component**               | **Weight** | **Purpose**                              |
|------------------------------|------------|------------------------------------------|
| Forward Displacement         | \(7.5\)    | Encourages forward motion.               |
| Sideways Penalty             | \(0.1\)    | Discourages drifting sideways.           |
| Velocity Reward              | \(5.0\)    | Rewards faster forward motion.           |
| Joint Velocity Penalty       | \(0.5\)    | Encourages smoother movements.           |
| Orientation Penalty (Roll)   | \(0.02\)   | Penalizes roll deviations.               |
| Orientation Penalty (Pitch)  | \(0.02\)   | Penalizes pitch deviations.              |
| Orientation Penalty (Yaw)    | \(0.05\)   | Penalizes yaw deviations (heaviest).     |
| Survival Reward              | \(1.0\)    | Rewards staying operational.             |

---

By integrating these components, the reward system encourages the Quadrobo robot to achieve efficient, stable, and purposeful locomotion.