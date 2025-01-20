### Models Documentation

This section describes the model architecture and configurations used for training and evaluating the quadruped robot in the TD3 (Twin Delayed Deep Deterministic Policy Gradient) framework.

---

#### **1. Model Configurations (`td3.yaml`)**
The configuration file specifies the hyperparameters for the TD3 model:

| Parameter              | Description                                                                                  | Default Value     |
|------------------------|----------------------------------------------------------------------------------------------|-------------------|
| `actor_hidden_sizes`   | Hidden layer sizes for the actor network.                                                    | `[256, 256]`      |
| `critic_hidden_sizes`  | Hidden layer sizes for the critic networks.                                                  | `[256, 256]`      |
| `learning_rate`        | Learning rate for the optimizer.                                                             | `0.001`           |
| `gamma`                | Discount factor for future rewards.                                                         | `0.99`            |
| `tau`                  | Soft update parameter for target networks.                                                  | `0.005`           |
| `policy_noise`         | Noise added to target policy during critic update.                                           | `0.2`             |
| `noise_clip`           | Range to clip noise during policy updates.                                                  | `0.5`             |
| `policy_freq`          | Frequency of delayed policy updates.                                                        | `2`               |
| `replay_buffer_size`   | Maximum size of the replay buffer.                                                           | `1_000_000`       |

---

#### **2. BaseModel Class**
The `BaseModel` class serves as an abstract base class for implementing reinforcement learning models. It defines the following methods:

- **`select_action`**: Select an action given the current state.
- **`train`**: Train the model using the replay buffer.
- **`save`**: Save the model weights to a file.
- **`load`**: Load the model weights from a file.

---

#### **3. Actor-Critic Networks**

##### **Actor Network**
The actor network is responsible for mapping states to actions. It uses a feedforward neural network with two hidden layers and a final `tanh` activation to ensure actions lie within the valid range.

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes):
        ...
```

##### **Critic Network**
The critic network evaluates the Q-value of a given state-action pair. It also uses two hidden layers and outputs a single scalar value.

```python
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        ...
```

---

#### **4. TD3 Implementation**
The `TD3` class implements the TD3 algorithm, inheriting from `BaseModel`. The key features include:

##### **Initialization**
- Actor and critic networks are initialized along with their target networks.
- Optimizers are created for the actor and both critics.

##### **Action Selection**
- Supports deterministic action selection during evaluation and exploration with noise during training.

```python
def select_action(self, state, deterministic=False):
    ...
```

##### **Training**
- **Critic Update**: Updates both critic networks to minimize the MSE loss between predicted Q-values and target Q-values.
- **Actor Update**: Updates the actor network less frequently (`policy_freq`) by maximizing Q-values for selected actions.
- **Soft Updates**: Updates target networks using the `tau` parameter.

```python
def train(self, replay_buffer):
    ...
```

##### **Checkpointing**
- Save and load methods are implemented for model weights.

```python
def save(self, path):
    ...
    
def load(self, path):
    ...
```

---

#### **5. Replay Buffer**
The replay buffer (`ReplayBuffer`) stores transitions `(state, action, reward, next_state, done)` for sampling during training. The replay buffer size is defined in the configuration file.

---

### Example Usage

##### **Training**
```bash
python scripts/train.py
```

##### **Evaluation**
```bash
python scripts/evaluate.py
```

The `TD3` model uses the actor to control the quadruped and trains the critic networks to provide accurate value estimates, enabling robust training and smooth control for the robot.