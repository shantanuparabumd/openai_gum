import os
import re
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from models.td3 import TD3
from environments.ant_env import AntEnv


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    print("Start Evaluation")

    cfg.environment.render_mode = cfg.evaluation.render_mode
    env = AntEnv(cfg.environment, cfg.rewards)

    model = TD3(cfg.model, env)

    # Load latest checkpoint
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

        while not done:
            action = model.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            episode_forces.append(state[-4:])  # Assuming the last 4 values are contact forces
            env.render()

        all_rewards.append(total_reward)
        contact_forces_log.append(episode_forces)
        print(f"Episode {episode + 1}/{cfg.evaluation.episodes}: Total Reward = {total_reward:.2f}")

    env.close()
    plot_contact_forces(contact_forces_log, cfg.logging.contact_forces_file)


def find_checkpoint_by_index(save_dir, index=None):
    """
    Finds a specific checkpoint by index or the latest checkpoint if index is None.

    Args:
        save_dir (str): Directory where checkpoints are stored.
        index (int, optional): Index of the desired checkpoint. If None, returns the latest.

    Returns:
        str: Path to the desired checkpoint or None if not found.
    """
    actor_checkpoints = [
        f for f in os.listdir(save_dir) if re.match(r"episode_(\d+)_actor\.pth$", f)
    ]
    if actor_checkpoints:
        actor_checkpoints.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))
        if index is None:
            # Return the latest checkpoint
            latest_checkpoint = actor_checkpoints[-1]
            print(f"Latest checkpoint: {latest_checkpoint}")
            return os.path.join(save_dir, latest_checkpoint.replace("_actor.pth", ""))
        elif 0 <= index < len(actor_checkpoints):
            # Return the checkpoint at the specified index
            specific_checkpoint = actor_checkpoints[index]
            print(f"Checkpoint at index {index}: {specific_checkpoint}")
            return os.path.join(save_dir, specific_checkpoint.replace("_actor.pth", ""))
        else:
            print(f"Index {index} out of range. Available checkpoints: {len(actor_checkpoints)}")
    else:
        print("No checkpoints found.")
    return None

def find_latest_checkpoint(save_dir):
    actor_checkpoints = [
        f for f in os.listdir(save_dir) if re.match(r"episode_(\d+)_actor\.pth$", f)
    ]
    if actor_checkpoints:
        latest_checkpoint = max(actor_checkpoints, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
        return os.path.join(save_dir, latest_checkpoint.replace("_actor.pth", ""))
    print(f"Latest Checkpoint {latest_checkpoint}")
    return None


def plot_contact_forces(contact_forces_log, contact_forces_file):
    forces = np.array(contact_forces_log).squeeze()
    time_steps = range(len(forces[0]))

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, forces[:, 0], label="Front Left Limb")
    plt.plot(time_steps, forces[:, 1], label="Front Right Limb")
    plt.plot(time_steps, forces[:, 2], label="Back Left Limb")
    plt.plot(time_steps, forces[:, 3], label="Back Right Limb")

    plt.xlabel("Time Steps")
    plt.ylabel("Ground Contact Force")
    plt.title("Limb Ground Contact Forces")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(contact_forces_file)
    print(f"Contact forces plot saved to {contact_forces_file}")
    plt.show()


if __name__ == "__main__":
    evaluate()
