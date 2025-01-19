import os
import re
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from models.td3 import TD3
from environments.quad_env import QuadEnv


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    print("Start Evaluation")

    cfg.environment.render_mode = cfg.evaluation.render_mode
    env = QuadEnv(cfg.environment, cfg.rewards)

    model = TD3(cfg.model, env)

    # Load latest checkpoint
    checkpoint_path = find_checkpoint_by_index(cfg.logging.save_dir,19000)
    checkpoint_path = find_latest_checkpoint(cfg.logging.save_dir)
    if checkpoint_path:
        print(f"Loading model from: {checkpoint_path}")
        model.load(checkpoint_path)
    else:
        print("No checkpoint found. Exiting...")
        return
    print("Checkpoints")
    all_rewards = []
    contact_forces_log = []
    

    for episode in range(cfg.evaluation.episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_forces = []
        done = False
        step_count = 0 
        while not done and step_count < cfg.evaluation.max_steps:
            action = model.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            episode_forces.append(state[-4:])  # Assuming the last 4 values are contact forces
            env.render()
            step_count += 1

        all_rewards.append(total_reward)
        contact_forces_log.append(episode_forces)
        print(f"Episode {episode + 1}/{cfg.evaluation.episodes}: Total Reward = {total_reward:.2f}")

    env.close()
    plot_contact_forces(contact_forces_log, cfg.logging.contact_forces_file)



def find_latest_checkpoint(save_dir):
    actor_checkpoints = [
        f for f in os.listdir(save_dir) if re.match(r"episode_(\d+)_actor\.pth$", f)
    ]
    if actor_checkpoints:
        latest_checkpoint = max(actor_checkpoints, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
        return os.path.join(save_dir, latest_checkpoint.replace("_actor.pth", ""))
    print(f"Latest Checkpoint {latest_checkpoint}")
    return None

def find_checkpoint_by_index(save_dir, index):
    """
    Find the checkpoint path for a specific episode index.

    Args:
        save_dir (str): Directory containing the checkpoints.
        index (int): Episode index to search for.

    Returns:
        str: Full path to the checkpoint if found, otherwise None.
    """
    actor_checkpoints = [
        f for f in os.listdir(save_dir) if re.match(r"episode_(\d+)_actor\.pth$", f)
    ]
    # Search for the checkpoint matching the specified index
    for checkpoint in actor_checkpoints:
        match = re.search(r"episode_(\d+)_actor\.pth$", checkpoint)
        if match and int(match.group(1)) == index:
            return os.path.join(save_dir, checkpoint.replace("_actor.pth", ""))
    
    print(f"No checkpoint found for index {index}")
    return None

def plot_contact_forces(contact_forces_log, contact_forces_file):
    """Plot contact forces for evaluation analysis."""
    # Determine the maximum sequence length
    max_length = max(len(episode) for episode in contact_forces_log)

    # Pad sequences to the same length
    padded_forces = np.array([
        np.pad(episode, ((0, max_length - len(episode)), (0, 0)), mode='constant', constant_values=0)
        for episode in contact_forces_log
    ])

    # Average contact forces across episodes
    avg_forces = np.mean(padded_forces, axis=0)

    # Plot contact forces
    time_steps = range(avg_forces.shape[0])
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, avg_forces[:, 0], label="Front Left Limb")
    plt.plot(time_steps, avg_forces[:, 1], label="Front Right Limb")
    plt.plot(time_steps, avg_forces[:, 2], label="Back Left Limb")
    plt.plot(time_steps, avg_forces[:, 3], label="Back Right Limb")

    plt.xlabel("Time Steps")
    plt.ylabel("Ground Contact Force")
    plt.title("Limb Ground Contact Forces")
    plt.legend()
    plt.grid()

    # Save the plot
    plt.tight_layout()
    plt.savefig(contact_forces_file)
    print(f"Contact forces plot saved to {contact_forces_file}")
    plt.show()
    
if __name__ == "__main__":
    evaluate()
