"""
Centralized configuration for the Game-Theoretic IDS with MARL.
All hyperparameters and settings are defined here.
"""

import os
import torch

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, CHECKPOINTS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# IoT Network Parameters
# ─────────────────────────────────────────────
NETWORK_SIZE = 20               # Number of IoT nodes (N)
NETWORK_TOPOLOGY = "random"     # "random", "scale_free", "grid", "star"
EDGE_PROBABILITY = 0.3          # For random (Erdos-Renyi) topology
INITIAL_ENERGY = 100.0          # Initial energy per node
ENERGY_MONITOR_COST = 0.5       # Energy cost for monitoring per step
ENERGY_DEEP_INSPECT_COST = 1.5  # Energy cost for deep inspection
ENERGY_IDLE_COST = 0.05         # Minimal energy for idle

# Dynamic Energy Model Coefficients
ALPHA_CPU = 1.0
BETA_COMM = 1.0
GAMMA_SENSOR = 1.0

# ─────────────────────────────────────────────
# Stackelberg Game Parameters
# ─────────────────────────────────────────────
# Defender parameters
DETECTION_REWARD_RANGE = (5.0, 15.0)    # R_i range per node
MONITORING_COST_RANGE = (1.0, 3.0)      # C_i range per node
DEFENDER_BUDGET = 8.0                    # Total monitoring budget B (sum x_i <= B)

# Attacker parameters
DAMAGE_RANGE = (5.0, 20.0)              # D_i range per node
DETECTION_PENALTY_RANGE = (8.0, 15.0)   # P_i range per node
ATTACKER_BUDGET = 5.0                   # Max simultaneous attacks

# ─────────────────────────────────────────────
# Environment Parameters
# ─────────────────────────────────────────────
STATE_DIM = 7                   # Features per agent observation
NUM_ACTIONS = 3                 # 0=idle, 1=monitor, 2=deep_inspect
MAX_STEPS_PER_EPISODE = 100     # Steps per episode
ATTACK_PROBABILITY = 0.3       # Base probability of attack per step (random attacker)
ALPHA_REWARD = 1.0              # Detection accuracy weight in reward
BETA_REWARD = 0.3               # Energy cost weight in reward
GAMMA_COOP = 0.1                # Cooperative neighbor bonus weight
KAPPA_DETERRENCE = 0.2          # Weight of the Stackelberg deterrence-threshold reward term

# Detection-component reward values. Missed attacks are penalized more
# heavily than false positives to push recall up under class imbalance
# (precision is already very high across methods; recall is the bottleneck).
MISSED_ATTACK_PENALTY = -1.2
FALSE_POSITIVE_PENALTY = -0.5
TRUE_POSITIVE_REWARD = 1.0
TRUE_NEGATIVE_REWARD = 0.2

# Fraction of attack samples to target when drawing training traffic
# (RealTrafficEnvironment only; evaluation always uses the natural,
# un-oversampled class distribution).
TRAIN_ATTACK_OVERSAMPLE_RATIO = 0.4

# ─────────────────────────────────────────────
# MARL Training Hyperparameters
# ─────────────────────────────────────────────
NUM_EPISODES = 1000             # Total training episodes
EVAL_INTERVAL = 50              # Evaluate every K episodes
CHECKPOINT_INTERVAL = 200       # Save checkpoint every K episodes
BATCH_SIZE = 64                 # Minibatch size
REPLAY_BUFFER_SIZE = 50000      # Experience replay capacity
GAMMA = 0.99                    # Discount factor
TAU = 0.005                     # Soft update coefficient

# Q-Learning / DQN specific
LEARNING_RATE_Q = 1e-3          # Learning rate for Q-learning/DQN
EPSILON_START = 1.0             # Initial exploration rate
EPSILON_END = 0.05              # Final exploration rate
EPSILON_DECAY = 0.985           # Epsilon decay per episode. Reaches the 0.05
                                # floor by ~episode 200 of a 300-episode run,
                                # leaving ~100 near-greedy fine-tuning episodes.
                                # (The old 0.995 never annealed: 0.995^300=0.22,
                                # i.e. agents were still 22% random at the end
                                # of training.)

# MADDPG specific
LEARNING_RATE_ACTOR = 1e-4      # Actor learning rate
LEARNING_RATE_CRITIC = 1e-3     # Critic learning rate
NOISE_SCALE = 0.1               # Ornstein-Uhlenbeck noise scale
NOISE_DECAY = 0.999             # Noise decay

# QMIX specific
LEARNING_RATE_QMIX = 5e-4      # QMIX learning rate
MIXING_EMBED_DIM = 32           # Mixing network embedding dimension

# ─────────────────────────────────────────────
# Network Architecture
# ─────────────────────────────────────────────
HIDDEN_DIM_1 = 128              # First hidden layer
HIDDEN_DIM_2 = 64               # Second hidden layer

# ─────────────────────────────────────────────
# Synthetic Traffic Generator
# ─────────────────────────────────────────────
NORMAL_TRAFFIC_RATE = 10.0      # Packets per second (Poisson lambda)
ATTACK_BURST_RATE = 50.0        # Attack burst rate
TRAFFIC_FEATURE_DIM = 10        # Number of traffic features

# ─────────────────────────────────────────────
# Experiment Settings
# ─────────────────────────────────────────────
ALGORITHMS = ["iql", "dqn", "maddpg", "qmix", "gnn_dqn"]
NETWORK_SIZES = [10, 20, 50, 100]    # For scalability analysis
ENERGY_BUDGETS = [50.0, 100.0, 200.0]  # For ablation
NUM_SEEDS = 5                   # Random seeds for averaging
SEEDS = [42, 123, 456, 789, 1011]

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
PLOT_DPI = 300                  # High-res for publication
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FONT_SIZE = 12
COLORS = {
    "iql": "#FF6B6B",
    "dqn": "#4ECDC4",
    "maddpg": "#45B7D1",
    "qmix": "#96CEB4",
    "baseline": "#CCCCCC",
}

# ─────────────────────────────────────────────
# Federated Learning & Privacy Parameters
# ─────────────────────────────────────────────
DP_ENABLED = True               # Enable Differential Privacy (clipping & noise)
DP_CLIP_NORM = 1.0              # DP weight update clipping threshold (sensitivity S)
DP_NOISE_SCALE = 0.01           # DP noise scale (sigma)
ENERGY_SAFETY_THRESHOLD = 20.0  # Min energy level to participate in federation
ADAPTIVE_FL_INTERVALS = {       # Dynamic federation intervals based on mean energy %
    "high": (75.0, 10),         # >=75% energy: federate every 10 episodes
    "medium": (50.0, 20),       # 50-75% energy: federate every 20 episodes
    "low": (25.0, 40),          # 25-50% energy: federate every 40 episodes
    "critical": (0.0, 80)       # <25% energy: federate every 80 episodes
}

