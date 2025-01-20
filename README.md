# Quadruped Robot Reinforcement Learning Framework

Welcome to the **Quadruped Robot Reinforcement Learning Framework**, a project that combines cutting-edge reinforcement learning techniques with MuJoCo physics simulation to train quadruped robots for stable and efficient locomotion. This framework allows you to experiment with custom environments, dynamic reward functions, and powerful actor-critic models to achieve state-of-the-art results.

---

## **Simulation Video**



https://github.com/user-attachments/assets/d51b1138-ef5c-40bf-964e-35f496e49969


---

## **Overview**

This project focuses on teaching a quadruped robot to walk using reinforcement learning techniques. It leverages the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm within a custom-designed MuJoCo-based environment (`QuadroboEnv`).

### Key Features:
- Custom OpenAI Gym-compatible environment.
- Dynamic reward function emphasizing forward motion, stability, and energy efficiency.
- TD3 model with tunable parameters.
- Logging, checkpointing, and evaluation for seamless experimentation.

---

## **Concept of the Project**

At its core, this project simulates a quadruped robot tasked with navigating a flat terrain. The robot learns to walk, balance, and optimize its gait through continuous interaction with the environment. A carefully designed reward system encourages the robot to move forward efficiently while penalizing energy wastage, unstable orientations, and off-track movements.

---

## **Folder Structure**

The project is well-organized into multiple modules for ease of development and scalability:

```plaintext
docs/
├── custom_env.md                 # Documentation for the custom environment
├── installation_and_troubleshoot.md  # Installation and troubleshooting steps
├── model_definition.md           # Details of the TD3 model implementation
├── project_structure.md          # Explanation of the project structure
├── reward_design.md              # Detailed breakdown of the reward function
└── training_and_evaluation.md    # Training and evaluation procedures
```

Refer to the **[Project Structure Documentation](docs/project_structure.md)** for a complete breakdown.

---

## **How to Use**

### 1. **Installation**
Refer to **[Installation and Troubleshooting Documentation](docs/installation_and_troubleshoot.md)** for detailed steps.

Quick start:
1. Install Miniconda and create a virtual environment:
   ```bash
   conda create -n mujoco_openai python=3.10
   conda activate mujoco_openai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the custom environment:
   ```bash
   pip install .
   ```

### 2. **Training**
Run the training script to teach the robot:
```bash
python scripts/train.py
```
Adjust the parameters in the `configs/` directory.

### 3. **Evaluation**
Evaluate a trained model:
```bash
python scripts/evaluate.py
```
Refer to the **[Training and Evaluation Documentation](docs/training_and_evaluation.md)** for more details.

---

## **Reward Design**

The reward function is designed to:
- Encourage forward motion.
- Penalize sideways drift and unstable orientation.
- Promote energy-efficient gait patterns.
- Reward the robot for maintaining a target velocity and balance.

For a comprehensive explanation, see the **[Reward Design Documentation](docs/reward_design.md)**.

---

## **Custom Environment**

The `QuadroboEnv` environment is the heart of this project. It is a MuJoCo-based environment designed to simulate the physics of a quadruped robot. The environment provides:
- Joint positions, velocities, and forces.
- Orientation feedback (roll, pitch, yaw).
- Contact forces for each leg.

Details are available in the **[Custom Environment Documentation](docs/custom_env.md)**.

---

## **TD3 Model**

The **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm is used to train the robot. The model features:
- Actor-critic networks for continuous action spaces.
- Noise injection for exploration.
- Soft updates to stabilize learning.

Read more in the **[Model Definition Documentation](docs/model_definition.md)**.


---


We welcome contributions and feedback. Feel free to open issues or submit pull requests to improve this project. 🚀
