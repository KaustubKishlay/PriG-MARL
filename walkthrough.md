# Walkthrough: Game-Theoretic IDS with MARL

## What Was Built

A complete, research-grade implementation of a **Game-Theoretic Intrusion Detection System** for IoT Networks using **Multi-Agent Reinforcement Learning** — 20 Python files across 8 modules, ready for IEEE paper experiments.

## Project Structure

```
M/
├── config.py                     # All hyperparameters
├── main.py                       # CLI entry point
├── run_experiments.py            # Batch experiment runner
├── requirements.txt
├── README.md
├── network/
│   └── iot_network.py            # IoT graph G=(V,E) with NetworkX
├── environment/
│   └── ids_env.py                # Multi-agent Gym-style environment
├── game/
│   ├── stackelberg.py            # Stackelberg bilevel optimization
│   └── nash.py                   # Nash equilibrium baseline
├── agents/
│   ├── replay_buffer.py          # Single + multi-agent buffers
│   ├── q_learning.py             # Independent Q-Learning (IQL)
│   ├── dqn.py                    # Double Dueling DQN
│   ├── maddpg.py                 # MADDPG (CTDE)
│   └── qmix.py                   # QMIX with hypernetwork mixer
├── data/
│   ├── dataset_loader.py         # NSL-KDD / UNSW-NB15 loaders
│   └── traffic_generator.py      # Synthetic IoT traffic (5 attack types)
├── training/
│   ├── trainer.py                # Unified MARL training loop
│   └── evaluator.py              # 5-metric evaluation framework
├── visualization/
│   ├── plots.py                  # 8 matplotlib publication plots
│   └── dashboard.py              # Interactive HTML dashboard
└── results/
    ├── figures/                   # 8 PNG plots (300 DPI)
    ├── tables/                    # CSV comparison tables
    ├── logs/                      # TensorBoard logs
    ├── checkpoints/               # Model checkpoints
    └── dashboard.html             # Interactive dashboard
```

## Smoke Test Results

**Configuration:** DQN, 10 nodes, 100 episodes, 50 steps/episode, strategic attacker

| Metric | Value |
|--------|-------|
| **Detection Accuracy** | 88.6% |
| **False Positive Rate** | 4.5% |
| **Precision** | 77.4% |
| **Recall** | 60.9% |
| **F1 Score** | 67.8% |
| **Network Lifetime** | 100% (no depletion) |

## Generated Plots

````carousel
![Training Reward Curves](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\training_curves.png)
<!-- slide -->
![Detection Accuracy](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\detection_accuracy.png)
<!-- slide -->
![Stackelberg Equilibrium](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\stackelberg_equilibrium.png)
<!-- slide -->
![Network Graph](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\network_graph.png)
<!-- slide -->
![FPR Comparison](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\fpr_comparison.png)
<!-- slide -->
![Convergence Comparison](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\convergence_comparison.png)
<!-- slide -->
![Energy Heatmap](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\energy_heatmap.png)
<!-- slide -->
![Confusion Matrix](C:\Users\kishu\.gemini\antigravity\brain\824c5b06-dca8-4c99-8805-a6dd87f5b84f\confusion_matrix.png)
````

## Key Implementation Details

### Stackelberg Game Solver
- Bilevel optimization with scipy SLSQP
- Attacker best response: closed-form threshold condition $x_i < D_i/(D_i + P_i)$
- 10 random restarts for global optimum

### MARL Algorithms
| Algorithm | Architecture | Key Feature |
|-----------|-------------|-------------|
| IQL | Per-node DQN | Fully decentralized |
| DQN | Double Dueling DQN | Reduced overestimation |
| MADDPG | Actor-Critic + OU noise | Centralized critic, continuous actions |
| QMIX | Hypernetwork mixer | Monotonic value decomposition |

### Environment Design
- **State:** 7-dim per agent (vulnerability, energy, traffic, importance, degree, attack status, neighbor threat)
- **Actions:** 3 discrete (idle/monitor/deep_inspect) with different detection probabilities (5%/65%/92%)
- **Reward:** α·detection - β·energy + γ·cooperation bonus

## How to Use

```bash
# Single algorithm
python main.py --algorithm maddpg --episodes 500 --network_size 20

# Full paper experiments (all 4 algorithms + ablations)
python run_experiments.py --experiment all

# Specific experiments
python run_experiments.py --experiment compare    # Algorithm comparison
python run_experiments.py --experiment attacker   # Attacker strategy comparison
python run_experiments.py --experiment ablation   # Network size ablation
```

## What Was Verified
- All dependencies install cleanly
- Full training pipeline runs end-to-end (DQN smoke test: 100 episodes)
- All 8 publication plots generated at 300 DPI
- Interactive HTML dashboard generated
- Results saved as JSON + CSV
- TensorBoard logging works
