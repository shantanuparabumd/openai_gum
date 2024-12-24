# Reinforcement Learning with TD3 for Ant Environment

This project demonstrates the use of the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm to train a simulated quadruped robot (Ant) in the Gymnasium environment. It is designed to showcase RL techniques and custom reward design for stable and efficient gait generation.

## What is TD3?
TD3 is a state-of-the-art actor-critic algorithm for continuous control tasks. It improves upon DDPG (Deep Deterministic Policy Gradient) by addressing overestimation bias in Q-value predictions. Key features include:
- **Target Policy Smoothing**: Adds noise to target actions to improve robustness.
- **Delayed Updates**: Updates the policy less frequently to stabilize training.
- **Clipped Double Q-Learning**: Uses two Q-networks to reduce overestimation bias.

This makes TD3 particularly effective for training agents in continuous action spaces, such as controlling a quadruped robot.

## What is Gymnasium (formerly OpenAI Gym)?
Gymnasium provides a standard interface for developing and comparing reinforcement learning algorithms. It offers:
- A variety of pre-built environments, including simulated robotics tasks.
- A simple API for environment interaction (`reset`, `step`, `render`).
- Tools for benchmarking RL agents.

In this project, we use the **Ant-v5** environment, which simulates a four-legged robot designed for locomotion tasks.

## Project Highlights
- **TD3 Algorithm**: Implements actor-critic networks with delayed updates for stable training.
- **Custom Rewards**: Encourages gait optimization, forward velocity, and torso stability.
- **Visualization**: Includes plots and videos to demonstrate training progress and evaluation.

---

## Features
- **Dynamic Gait Learning**: Optimizes the Ant robot for forward motion with a stable trot gait.
- **Configurable Parameters**: Easy-to-tune hyperparameters and reward components.
- **Rendering Modes**: Visualizes training in `rgb_array` mode and evaluation in `human` mode.

---

## Examples

### **Training Progress**
Training the agent to maximize forward velocity while maintaining stability:

![Training Metrics](./outputs/metrics.png)


### **Gait Plotting**

Plotting og the contact forces to understand the gait. 

![Training Metrics](./outputs/contact_forces.png)

### **Evaluation Visualization**
The trained agent demonstrating a trot gait during evaluation:


---

## Configuration Files
All configurations for the project are managed through YAML files, allowing easy tuning of parameters and modularity:

### **`configs/config.yaml`**
- Main configuration file.
- Defines training, evaluation, and logging settings.

### **`configs/environment/ant.yaml`**
- Environment-specific parameters for Ant-v5, including rewards and dynamics.

### **`configs/model/td3.yaml`**
- Hyperparameters for the TD3 algorithm, such as learning rate, noise, and policy frequency.

### **`configs/rewards/rewards.yaml`**
- Custom reward components for gait optimization, stability, and efficiency.

These files ensure that the project is flexible and easy to customize for different tasks.

---

## How to Run

### Setup

1. Clone the repository:
   ```bash
   git clone openai_gym
   cd openai_gym
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n rl_env python=3.8 -y
   conda activate rl_env
   ```

3. Install dependencies:
   ```bash
   conda install --file requirements.txt
   ```

4. Configure the environment using `configs/config.yaml`.

### Training
Run the training script to train the TD3 agent:
```bash
python scripts/train.py
```

### Evaluation
Evaluate the trained agent:
```bash
python scripts/evaluate.py
```

---


