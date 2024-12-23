import numpy as np
from typing import Dict


def calculate_reward(observation: np.ndarray, reward: float, reward_config: Dict) -> float:
    """
    Augment the environment's reward with penalties for gait, sway, and height.

    Args:
        observation (np.ndarray): The observation from the environment.
        reward (float): The default reward from the environment.
        reward_config (Dict): Configuration for custom reward components.

    Returns:
        float: Total reward for the step.
    """
    # Start with the default reward
    total_reward = reward

    # Ensure observation has sufficient dimensions
    if len(observation) < 17:  # Adjust if observation dimensions change
        raise ValueError(f"Observation size {len(observation)} is too small for reward calculation.")

    # Extract observations
    forward_velocity = observation[13]  # Forward velocity (qvel[0])
    sideways_velocity = observation[14]  # Sideways velocity (qvel[1])
    torso_height = observation[0]  # Torso height (qpos[2])
    contact_forces = observation[-4:]  # Last 4 values correspond to cfrc_ext for 4 legs

    # Forward velocity reward
    velocity_target = reward_config["forward_velocity"]["target"]
    velocity_error = abs(forward_velocity - velocity_target)
    total_reward += reward_config["forward_velocity"]["weight"] * (1 - velocity_error)

    # Sideways sway penalty
    sway_penalty = abs(sideways_velocity)
    total_reward -= reward_config["sideways_sway"]["weight"] * sway_penalty

    # Torso height penalty
    min_height, max_height = reward_config["torso_height"]["range"]
    if not (min_height <= torso_height <= max_height):
        total_reward += reward_config["torso_height"]["penalty"]

    # Trot gait penalty
    # Front-left (0) <-> Back-right (2), Front-right (1) <-> Back-left (3)
    trot_penalty = abs(contact_forces[0] - contact_forces[2]) + abs(contact_forces[1] - contact_forces[3])
    total_reward -= reward_config["trot_gait"]["weight"] * trot_penalty

    # Optional debug information
    debug_info = {
        "forward_velocity": forward_velocity,
        "sideways_velocity": sideways_velocity,
        "torso_height": torso_height,
        "trot_penalty": trot_penalty,
        "total_reward": total_reward,
    }
    print_debug_info(debug_info, reward_config.get("debug", False))

    return total_reward


def print_debug_info(debug_info: Dict, debug: bool) -> None:
    """
    Print debug information if debugging is enabled.

    Args:
        debug_info (Dict): Dictionary of debug information to print.
        debug (bool): Whether to print debug information.
    """
    if debug:
        print("Debug Info:")
        for key, value in debug_info.items():
            print(f"  {key}: {value:.4f}")
