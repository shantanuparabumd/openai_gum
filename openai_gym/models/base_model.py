import abc

class BaseModel(abc.ABC):
    def __init__(self, config, env):
        self.config = config
        self.env = env

    @abc.abstractmethod
    def select_action(self, state):
        """Select an action given the state."""
        pass

    @abc.abstractmethod
    def train(self, replay_buffer):
        """Train the model using the replay buffer."""
        pass

    @abc.abstractmethod
    def save(self, path):
        """Save the model to a file."""
        pass

    @abc.abstractmethod
    def load(self, path):
        """Load the model from a file."""
        pass
