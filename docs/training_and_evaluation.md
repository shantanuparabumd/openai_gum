### Training and Evaluation Guide

#### **Training**
The `train.py` script is used to train the model using the configuration provided in `configs/config.yaml`. The script performs several tasks, including initializing the environment, creating the model, and training with checkpointing. Below are the steps to run and understand the training process:

1. **Key Configuration Parameters for Training:**
   - **`episodes`**: Total number of training episodes.
   - **`max_steps`**: Maximum steps per episode.
   - **`render_mode`**: Rendering mode during training (e.g., `"rgb_array"` or `None` for performance).
   - **`resume`**: Resume training from the latest checkpoint if set to `true`.
   - **`overwrite`**: If set to `true`, overwrite existing checkpoints and start training afresh.
   - **`warmup_steps`**: Number of random actions used to populate the replay buffer.

2. **Run Training Script:**
   ```bash
   python scripts/train.py
   ```

   - Checkpoints are saved periodically in the directory specified by `logging.save_dir`.
   - Training metrics (e.g., rewards and losses) are logged and saved as plots.

3. **Logging and Metrics:**
   - **Logs**: Training progress logs are stored in `outputs/<timestamp>/train.log`.
   - **Metrics**: Reward and loss metrics are saved as `outputs/metrics.png`.

---

#### **Evaluation**
The `evaluate.py` script is used to evaluate the trained model and analyze its performance. Evaluation focuses on metrics like rewards, contact forces, and overall behavior.

1. **Key Configuration Parameters for Evaluation:**
   - **`episodes`**: Number of episodes to evaluate.
   - **`max_steps`**: Maximum steps per episode during evaluation.
   - **`render_mode`**: Rendering mode for evaluation (e.g., `"human"` for visualization).
   - **`contact_forces_file`**: Path to save the contact forces plot.

2. **Run Evaluation Script:**
   ```bash
   python scripts/evaluate.py
   ```

   - The script loads the latest checkpoint automatically or a specific checkpoint if provided.
   - Evaluation metrics like rewards and contact forces are logged.

3. **Outputs from Evaluation:**
   - **Rewards**: Total rewards for each evaluation episode are printed.
   - **Contact Forces**: A plot of ground contact forces across limbs is saved as `outputs/contact_forces.png`.
   - **Visualization**: If `render_mode` is set to `"human"`, the robot's behavior is rendered in real time.

---

#### **Key Functions**

1. **Checkpoints Handling:**
   - `handle_checkpoints`: Manages resuming or overwriting training checkpoints.
   - `find_latest_checkpoint`: Automatically identifies the latest checkpoint based on the episode index.
   - `find_checkpoint_by_index`: Finds a checkpoint corresponding to a specific episode.

2. **Plotting Utilities:**
   - `plot_metrics`: Visualizes reward and loss trends during training.
   - `plot_contact_forces`: Generates contact force plots for evaluation analysis.

---

#### **Tips**
- Adjust `max_steps` and `episodes` for faster or more thorough training/evaluation.
- Use `logging.checkpoint_freq` to control checkpoint intervals during training.
- For debugging, set `render_mode` to `"human"` to visualize the robot’s behavior. 

This modular setup ensures flexibility and ease of use for both training and evaluation.