# Game-Theoretic Intrusion Detection in IoT Networks using MARL

A research-grade implementation of a **Game-Theoretic Intrusion Detection System (IDS)** for IoT networks using **Multi-Agent Reinforcement Learning (MARL)**. Publication-ready for IEEE/Springer journals.

## Overview

This project models the interaction between a **Defender (IDS)** and an **Attacker** as a **Stackelberg Game**, with distributed IoT nodes acting as cooperative MARL agents.

**Novel Contributions:**
1. Hybrid model combining Stackelberg Game Theory + MARL
2. Energy-aware IDS optimized for resource-constrained IoT
3. Distributed learning with no central controller
4. Adaptive attacker modeling via game-theoretic best response

## Architecture

```
main.py / run_experiments.py / run_publication.py
├── training/
│   ├── trainer.py               # Unified MARL training loop
│   └── evaluator.py             # 5-metric evaluation framework
├── agents/
│   ├── q_learning.py            # Independent Q-Learning (IQL)
│   ├── dqn.py                   # Double Dueling DQN
│   ├── maddpg.py                # MADDPG (CTDE)
│   ├── qmix.py                  # QMIX with hypernetwork mixer
│   ├── baselines.py             # 5 baseline methods for comparison
│   └── replay_buffer.py         # Experience replay buffers
├── game/
│   ├── stackelberg.py           # Stackelberg bilevel optimization
│   └── nash.py                  # Nash equilibrium baseline
├── environment/
│   └── ids_env.py               # Multi-agent Gym-style environment
├── network/
│   └── iot_network.py           # IoT graph G=(V,E) model
├── data/
│   ├── dataset_loader.py        # NSL-KDD / UNSW-NB15 loaders
│   └── traffic_generator.py     # Synthetic IoT traffic (5 attack types)
├── visualization/
│   ├── plots.py                 # 8 matplotlib publication plots (300 DPI)
│   └── dashboard.py             # Interactive HTML dashboard
├── paper/
│   └── paper.tex                # IEEE LaTeX paper template
└── results/                     # All outputs (figures, tables, logs)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Single Algorithm Training
```bash
python main.py --algorithm dqn --episodes 500 --network_size 20
python main.py --algorithm maddpg --episodes 1000 --attacker stackelberg
python main.py --algorithm qmix --episodes 500 --topology scale_free
```

### Full Publication Experiments (Recommended)
```bash
# Full suite: 4 MARL algorithms + 5 baselines + 4 ablation studies
python run_publication.py

# Quick test mode (fewer episodes)
python run_publication.py --quick
```

### Individual Experiments
```bash
python run_experiments.py --experiment compare    # Algorithm comparison
python run_experiments.py --experiment attacker   # Attacker strategy comparison
python run_experiments.py --experiment ablation   # Network size ablation
```

## MARL Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **IQL** | Decentralized | Independent Q-Learning per node |
| **DQN** | Decentralized | Double Dueling DQN per node |
| **MADDPG** | CTDE | Centralized critic, decentralized actors |
| **QMIX** | CTDE | Value decomposition with mixing network |

## Baseline Methods (for paper comparison)

These are the baselines actually wired into the real-dataset comparison in
`run_real_dataset.py` (the source of truth for `results/real_dataset_*.json`
and the paper's tables):

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Random Policy** | Uniform random actions | Lower bound |
| **Always Monitor** | Maximum monitoring | Upper bound (energy wasteful) |
| **Static Stackelberg** | Fixed game-theoretic equilibrium policy | Tests if learning is needed |
| **Adaptive Stackelberg** | Equilibrium re-solved from attack history | Tests if reactive game theory is enough |
| **LSTM / Autoencoder / CNN / Transformer** | Centralized deep-learning IDS classifiers | Tests if MARL adds value over centralized DL |

Note: `agents/baselines.py` also defines a `RandomForestIDS` baseline, but it
is not currently wired into the real-dataset pipeline (it trains on the
synthetic traffic generator, not on NSL-KDD/UNSW-NB15/CICIDS2017) -- it only
appears in the synthetic-environment comparison run by `run_publication.py`.

## Game-Theoretic Model

- **Stackelberg Game**: Defender (leader) sets monitoring policy, Attacker (follower) best-responds
- **Bilevel Optimization**: scipy SLSQP solver with multiple restarts
- **Attacker Best Response**: Closed-form threshold: attack node i iff x_i < D_i/(D_i + P_i)

## Evaluation Metrics

1. **Detection Accuracy** — (TP + TN) / Total
2. **False Positive Rate** — FP / (FP + TN)
3. **Energy Consumption** — Cumulative per episode
4. **Network Lifetime** — Ratio of surviving nodes
5. **Convergence Speed** — Episodes to 95% of final reward

## Outputs

- `results/figures/` — 12+ publication-quality plots (300 DPI)
- `results/tables/` — Algorithm comparison CSVs
- `results/logs/` — TensorBoard training logs
- `results/dashboard.html` — Interactive visualization
- `results/checkpoints/` — Model checkpoints
- `paper/paper.tex` — IEEE LaTeX paper template

## Paper Structure (IEEE Format)

| Section | Source |
|---------|--------|
| III. System Model | `network/iot_network.py` |
| IV. Game Formulation | `game/stackelberg.py` |
| V. MARL Framework | `agents/` |
| VI. Algorithm Design | `training/trainer.py` |
| VII. Results | `results/tables/*.csv` + `results/figures/*.png` |

## Configuration

All hyperparameters in `config.py`: network size, topology, energy budgets, game parameters, learning rates, epsilon schedules, buffer sizes, visualization settings.
