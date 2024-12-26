import gymnasium as gym
import environments  # Ensure this imports your custom environment

# Load the custom environment
env = gym.make("QuadRobot-v0", render_mode="human")

# Test environment interaction
observation, _ = env.reset()
done = False
step = 0
while step < 1000:
    print(f"step {step}")
    action = env.action_space.sample()  # Take random actions
    observation, reward, terminated, truncated, _ = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}")
    done = terminated or truncated
    env.render()
    step += 1

env.close()
