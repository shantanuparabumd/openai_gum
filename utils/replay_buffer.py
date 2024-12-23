import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        """Initialize the replay buffer.

        Args:
            max_size (int): Maximum size of the buffer.
            state_dim (int): Dimension of the state.
            action_dim (int): Dimension of the action.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Preallocate memory for buffer
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add a new experience to the buffer.

        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode terminated.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = 64) -> tuple:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of samples to retrieve.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, next states, and dones.
        """
        assert self.size >= batch_size, "Not enough samples in buffer to sample."
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
