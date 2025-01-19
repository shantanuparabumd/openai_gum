import gymnasium as gym
import environments  # Ensure this imports your custom environment
import time


# Load the custom environment
env = gym.make("QuadRobot-v0", render_mode="human")

# Test the environment
observation, _ = env.reset()
done = False
step = 0

while step < 100000:
    print(f"Step: {step}")
    action = env.action_space.sample()  # Random actions
    observation, reward, terminated, truncated, _ = env.step(action)
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    done = terminated or truncated
    step += 1

    # Sleep to reduce rendering frequency
    # time.sleep(1 / env.metadata["render_fps"])


env.close()
