import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.base_model import BaseModel
from utils.replay_buffer import ReplayBuffer


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.layers(state)


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.layers(x)


class TD3(BaseModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device Used ", self.device)
        # Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim, max_action, config.actor_hidden_sizes).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, config.actor_hidden_sizes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)

        self.critic1 = Critic(state_dim, action_dim, config.critic_hidden_sizes).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim, config.critic_hidden_sizes).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.learning_rate)

        self.critic2 = Critic(state_dim, action_dim, config.critic_hidden_sizes).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, config.critic_hidden_sizes).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.learning_rate)

        # TD3 Hyperparameters
        self.max_action = max_action
        self.tau = config.tau
        self.gamma = config.gamma
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_freq = config.policy_freq
        self.total_it = 0

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        if deterministic:
            return self.actor(state).cpu().data.numpy()
        else:
            action = self.actor(state).cpu().data.numpy()
            action += np.random.normal(0, self.max_action * self.policy_noise, size=action.shape)
            return action.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Critic Update
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed Actor Update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            self.soft_update(self.actor, self.actor_target)

        return critic1_loss.item(), critic2_loss.item()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic1.state_dict(), f"{path}_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{path}_critic2.pth")
        print(f"Checkpoint saved at {path}")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth", map_location=self.device))
        self.critic1.load_state_dict(torch.load(f"{path}_critic1.pth", map_location=self.device))
        self.critic2.load_state_dict(torch.load(f"{path}_critic2.pth", map_location=self.device))
        print(f"Model loaded from: {path}")
