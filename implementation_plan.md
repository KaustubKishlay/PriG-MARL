# Game-Theoretic Intrusion Detection in IoT Networks using MARL

## Goal

Build a **complete, research-grade Python/PyTorch implementation** of a Game-Theoretic Intrusion Detection System (IDS) for IoT networks using Multi-Agent Reinforcement Learning (MARL). The system models the interaction between a **Defender (IDS)** and an **Attacker** as a **Stackelberg Game**, with distributed IoT nodes acting as MARL agents. The codebase will be directly usable for IEEE paper experiments.

All code goes into `c:\Users\kishu\Desktop\M`.

---

## Proposed Changes

### Phase 1 — Project Structure & Configuration

#### [NEW] `requirements.txt`
- PyTorch, NumPy, Pandas, Matplotlib, Seaborn, NetworkX, Scikit-learn, tqdm, tensorboard

#### [NEW] `config.py`
- All hyperparameters in one place: network size (N=20), learning rates, gamma, alpha/beta weights, energy budgets, episode counts, Stackelberg game parameters (R_i, C_i, D_i, P_i)

---

### Phase 2 — Network & Environment Model

#### [NEW] `network/iot_network.py`
- `IoTNetwork` class: Graph $G = (V, E)$ with N nodes
- Uses NetworkX to generate topology (random, scale-free, or grid)
- Node attributes: `traffic_rate`, `vulnerability_score`, `energy_level`, `is_compromised`
- Edge attributes: `bandwidth`, `latency`

#### [NEW] `environment/ids_env.py`
- Custom multi-agent environment (Gym-style API: `reset()`, `step()`, `observe()`)
- **State space** per agent: `[traffic_i, alerts_i, energy_i, neighbor_status, attack_history_i]` — continuous vector
- **Action space** per agent: `{0: idle, 1: monitor, 2: deep_inspect}` — discrete
- **Reward function**: $r_i^t = \alpha \cdot \text{Detection}_i - \beta \cdot \text{Energy}_i + \gamma_{coop} \cdot \text{Neighbor Bonus}$
- Implements attack injection: random / strategic (Stackelberg follower)
- Tracks global metrics: detection rate, FPR, energy consumption per step

---

### Phase 3 — Game-Theoretic Layer (Stackelberg Game)

#### [NEW] `game/stackelberg.py`
- **Defender utility**: $U_D(\mathbf{x}, \mathbf{a}) = \sum_{i=1}^{N} [R_i \cdot x_i \cdot a_i - C_i \cdot x_i]$
- **Attacker utility**: $U_A(\mathbf{x}, \mathbf{a}) = \sum_{i=1}^{N} [D_i \cdot a_i \cdot (1 - x_i) - P_i \cdot a_i \cdot x_i]$
- `AttackerBestResponse(x)` — closed-form: attack node $i$ iff $D_i(1-x_i) > P_i \cdot x_i$
- `DefenderOptimize(budget)` — scipy optimize over $\mathbf{x}$ subject to $\sum x_i \leq B$
- `StackelbergEquilibrium` — solve bilevel optimization
- Returns equilibrium strategies $(\mathbf{x}^*, \mathbf{a}^*)$

#### [NEW] `game/nash.py`
- Nash equilibrium solver (for comparison baseline)
- Mixed-strategy computation using linear programming

---

### Phase 4 — MARL Agents

#### [NEW] `agents/q_learning.py`
- **Independent Q-Learning (IQL)** agent per node
- Tabular Q-table (discretized states) or DQN with small MLP
- Epsilon-greedy exploration → decaying schedule
- Bellman update: $Q(s,a) \leftarrow Q(s,a) + \eta[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

#### [NEW] `agents/dqn.py`
- Deep Q-Network agent with:
  - 3-layer MLP (128→64→|A|)
  - Replay buffer (size 50K)
  - Target network (soft update τ=0.005)
  - Huber loss

#### [NEW] `agents/maddpg.py`
- **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient)
- Centralized training, decentralized execution (CTDE)
- Actor: local observation → action (continuous monitoring intensity)
- Critic: all observations + all actions → Q-value
- Soft actor-critic style updates
- Ornstein-Uhlenbeck noise for exploration

#### [NEW] `agents/qmix.py`
- **QMIX** value decomposition
- Individual agent Q-networks
- Mixing network with hypernetwork (state-conditioned monotonic combination)
- Centralized training with joint reward

#### [NEW] `agents/replay_buffer.py`
- Shared replay buffer for experience storage
- Support for per-agent and joint transitions

---

### Phase 5 — Dataset Integration

#### [NEW] `data/dataset_loader.py`
- Load NSL-KDD (`KDDTrain+.txt`, `KDDTest+.txt`)
- Load UNSW-NB15 (`UNSW-NB15_1.csv` … `UNSW-NB15_4.csv`)
- Preprocessing pipeline:
  - Label encoding for categorical features (`protocol_type`, `service`, `flag`)
  - StandardScaler normalization
  - Binary label mapping (normal=0, attack=1)
  - Multi-class attack categorization
- Feature selection (top-K by mutual information)
- Train/val/test splits

#### [NEW] `data/traffic_generator.py`
- Synthetic IoT traffic generator (when real datasets unavailable)
- Normal traffic: Poisson-distributed packet arrivals
- Attack traffic: burst patterns, scan patterns, DoS floods
- Maps to same feature format as NSL-KDD

---

### Phase 6 — Training Pipeline

#### [NEW] `training/trainer.py`
- `MARLTrainer` class orchestrating the full loop:
  1. Reset environment
  2. Each agent observes local state
  3. Select actions (from MARL policy)
  4. Stackelberg attacker selects targets (best response to current defense)
  5. Environment steps → rewards, next states
  6. Store transitions, update networks
  7. Log metrics to TensorBoard
- Supports: IQL, DQN, MADDPG, QMIX (selectable via config)
- Checkpoint saving every K episodes

#### [NEW] `training/evaluator.py`
- Evaluation loop (no exploration)
- Computes all 5 required metrics:
  - **Detection Accuracy** (TP+TN)/(Total)
  - **False Positive Rate** FP/(FP+TN)
  - **Energy Consumption** (cumulative per episode)
  - **Network Lifetime** (episodes until first node depleted)
  - **Convergence Speed** (episodes to 95% of final reward)

---

### Phase 7 — Visualization & Results

#### [NEW] `visualization/plots.py`
- Publication-quality plots (matplotlib, IEEE style):
  1. **Training curves**: reward vs episodes (all algorithms overlaid)
  2. **Detection accuracy** over time
  3. **FPR comparison** bar chart
  4. **Energy consumption** heatmap over network topology
  5. **Stackelberg equilibrium** visualization (defender vs attacker strategies)
  6. **Network graph** with node colors (attacked/monitored/normal)
  7. **Convergence comparison** across IQL, DQN, MADDPG, QMIX
  8. **Confusion matrix** for detection performance

#### [NEW] `visualization/dashboard.py`
- Interactive HTML dashboard (single-page) showing:
  - Live network topology with attack/defense overlay
  - Metric gauges (accuracy, FPR, energy)
  - Training progress charts

---

### Phase 8 — Main Entry Points

#### [NEW] `main.py`
- CLI entry point: `python main.py --algorithm maddpg --episodes 1000 --network_size 20`
- Runs full pipeline: env setup → training → evaluation → plot generation
- Saves all results to `results/` directory

#### [NEW] `run_experiments.py`
- Batch experiment runner for paper:
  - Compare: IQL vs DQN vs MADDPG vs QMIX
  - Compare: With vs without Stackelberg (random attacker baseline)
  - Ablation: varying N (10, 20, 50 nodes)
  - Ablation: varying energy budget
- Generates all tables and figures for the paper

#### [NEW] `README.md`
- Project overview, installation, usage, paper citation

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                    main.py / run_experiments.py       │
├──────────────────────────────────────────────────────┤
│                   training/trainer.py                 │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────┐ │
│  │ MARL Agents  │  │  Environment   │  │Stackelberg│ │
│  │ IQL/DQN/     │←→│  ids_env.py    │←→│  Game     │ │
│  │ MADDPG/QMIX │  │                │  │  Solver   │ │
│  └─────────────┘  └────────────────┘  └───────────┘ │
│         ↑                  ↑                         │
│  ┌─────────────┐  ┌────────────────┐                 │
│  │Replay Buffer│  │  IoT Network   │                 │
│  └─────────────┘  │  (NetworkX)    │                 │
│                   └────────────────┘                  │
│                          ↑                            │
│                   ┌────────────────┐                  │
│                   │ Dataset Loader │                  │
│                   │ NSL-KDD /      │                  │
│                   │ UNSW-NB15 /    │                  │
│                   │ Synthetic      │                  │
│                   └────────────────┘                  │
├──────────────────────────────────────────────────────┤
│  training/evaluator.py  │  visualization/plots.py    │
│  → 5 key metrics        │  → 8 publication plots     │
└──────────────────────────────────────────────────────┘
```

---

## Key Mathematical Implementation Details

### Stackelberg Equilibrium Solver
```
Algorithm: Bilevel Optimization for Stackelberg IDS

Input: Node params {R_i, C_i, D_i, P_i}, budget B
Output: Equilibrium (x*, a*)

1. Define attacker best response:
   For each node i:
     a_i*(x) = 1  if  D_i(1 - x_i) > P_i·x_i
     a_i*(x) = 0  otherwise
     Threshold: x_i < D_i / (D_i + P_i)

2. Solve defender problem:
   max_x  Σ_i [R_i · x_i · a_i*(x) - C_i · x_i]
   s.t.   Σ_i x_i ≤ B
          0 ≤ x_i ≤ 1  ∀i

3. Use scipy.optimize.minimize (SLSQP) with
   attacker best response embedded in objective
```

### MADDPG Training Loop
```
For each episode:
  For each timestep t:
    For each agent i:
      a_i = μ_θi(o_i) + noise
    Execute actions, get rewards r, next obs o'
    Store (o, a, r, o') in replay buffer D
    
    For each agent i:
      Sample minibatch from D
      y = r_i + γ · Q_φi'(o', a')|a'=μ'(o')
      Update critic: minimize (Q_φi(o, a) - y)²
      Update actor: ∇_θi J = ∇_a Q_φi(o,a)|a=μ(o) · ∇_θi μ_θi(o_i)
      Soft update targets: θ' ← τθ + (1-τ)θ'
```

---

## Verification Plan

### Automated Tests
1. **Unit tests**: Stackelberg solver finds known equilibrium on 2-node toy game
2. **Environment tests**: Verify reward computation, state transitions, energy depletion
3. **Training smoke test**: 100 episodes of IQL on 5-node network → reward increases
4. **Full experiment**: Run all 4 algorithms for 500+ episodes on 20-node network

### Validation Metrics (must match paper claims)
| Metric | Target Range |
|--------|-------------|
| Detection Accuracy | 85–95% |
| False Positive Rate | < 10% |
| Energy Savings vs naive | 15–30% |
| Convergence (MADDPG) | < 300 episodes |

### Output Verification
- All 8 plot types generated in `results/figures/`
- Comparison tables saved as CSV in `results/tables/`
- TensorBoard logs in `results/logs/`
- Interactive dashboard accessible in browser

---

## Open Questions

> [!IMPORTANT]
> **Dataset availability**: Do you have NSL-KDD or UNSW-NB15 datasets downloaded locally, or should I use the synthetic traffic generator only?

> [!IMPORTANT]
> **Compute resources**: MADDPG/QMIX training on 50-node networks can be slow on CPU. Do you have GPU (CUDA) available?

> [!NOTE]
> **Network topology**: The default is a random graph (Erdős–Rényi). Would you prefer a specific IoT topology (star, mesh, scale-free)?

> [!NOTE]
> **Interactive dashboard**: Should I build the HTML visualization dashboard, or are matplotlib plots sufficient for the paper?
