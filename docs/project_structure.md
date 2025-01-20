### **Project Folder Structure and Overview**

#### **1. `configs/`**
Contains configuration files to customize the environment, model, and reward settings.

- **Files**:
  - `config.yaml`: Main configuration file that combines all settings.
  - **Subfolders**:
    - `environment/`: Contains specific configurations for environments (`ant.yaml`, `quadrobo.yaml`).
    - `model/`: Defines hyperparameters for models (`td3.yaml`).
    - `rewards/`: Reward shaping configurations (`rewards.yaml`).

#### **2. `environments/`**
Contains the implementation of custom MuJoCo environments.

- **Files**:
  - `ant_env.py`: Code for the Ant environment.
  - `quad_env.py`: Code for the Quadrobo environment.
- **Purpose**:
  - Provides the simulation logic and state transitions for each environment.

#### **3. `models/`**
Implements RL algorithms and their base functionalities.

- **Files**:
  - `td3.py`: Implements the TD3 algorithm.
  - `base_model.py`: Base class with common functionality like saving and loading weights.

#### **4. `utils/`**
Utility scripts for shared functionality.

- **Files**:
  - `replay_buffer.py`: Implements the replay buffer for experience storage.
  - `rewards.py`: Custom reward functions used during training.

#### **5. `scripts/`**
Scripts for running training, evaluation, and testing.

- **Files**:
  - `train.py`: Launches the training process.
  - `evaluate.py`: Evaluates the trained model.
  - `test_env.py`: Tests and debugs environments.
  - `view_model.py`: Visualizes the model.

#### **6. `outputs/`**
Stores logs, metrics, and checkpoints generated during training and evaluation.

- **Structure**:
  - Organized by timestamps.
  - Contains:
    - Training logs (`train.log`).
    - Evaluation logs (`evaluate.log`).
    - Model checkpoints.
    - Metrics and visualizations (`metrics.png`, `contact_forces.png`).

#### **7. `quadrobo_gym/`**
Implements and registers the Quadrobo custom Gym environment.

- **Files**:
  - `env.py`: Defines the `QuadroboEnv` class.
  - `setup.py`: Setup script to register the environment.
  - `register.py`: Handles registration logic.

#### **8. `videos/`**
Stores recorded videos of evaluations for visualization (`evaluation_demo.webm`).

#### **9. `logs/`**
Holds debugging and runtime logs.

---

### **Key Features**
1. **Modularity**: Each folder handles a specific aspect (environments, models, scripts, etc.), making it easy to update or extend.
2. **Configuration-Driven**: Centralized configurations allow seamless customization without changing code.
3. **Outputs and Logs**: Clear separation for logs, metrics, and checkpoints for better experiment tracking.

This simplified structure ensures flexibility, readability, and ease of use for development and experimentation.