"""
Run MARL IDS experiments on REAL NSL-KDD dataset.

This is the key experiment for publication:
- Uses actual network intrusion traffic (125K+ labeled samples)
- Tests all 4 MARL algorithms + baselines on real data
- Generates comparison tables and plots

Usage:
    python run_real_dataset.py
    python run_real_dataset.py --algorithm maddpg --episodes 300
"""

import os, sys, json, time, argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from network.iot_network import IoTNetwork
from data.real_dataset import NSLKDDProcessor, UNSWNB15Processor, CICIDS2017Processor, RealTrafficEnvironment
from game.stackelberg import create_game_from_network, AdaptiveStackelbergGame
from agents.q_learning import IQLAgent
from agents.dqn import DQNAgent
from agents.maddpg import MADDPGAgent
from agents.qmix import QMIXAgent
from agents.federated import FederatedServer
from agents.baselines import (
    RandomPolicy, AlwaysMonitorPolicy, StaticStackelbergPolicy, AdaptiveStackelbergPolicy,
    RandomForestIDS, evaluate_baseline,
)
from agents.deep_baselines import LSTM_Baseline, Autoencoder_Baseline, CNN_Baseline, Transformer_Baseline
from visualization.plots import (
    setup_style, plot_training_curves, plot_fpr_comparison,
    plot_confusion_matrix, plot_convergence_comparison,
)
from visualization.xai_explainer import explain_real_dataset_agent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def collect_run_metadata(seeds):
    """Environment/config snapshot saved alongside results for reproducibility."""
    import platform
    import importlib.metadata as md
    versions = {}
    for pkg in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "networkx", "shap"]:
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": versions,
        "seeds": list(seeds),
        "config_snapshot": {
            k: getattr(config, k)
            for k in [
                "NETWORK_SIZE", "NETWORK_TOPOLOGY", "MAX_STEPS_PER_EPISODE",
                "ALPHA_REWARD", "BETA_REWARD", "GAMMA_COOP", "KAPPA_DETERRENCE",
                "MISSED_ATTACK_PENALTY", "FALSE_POSITIVE_PENALTY",
                "TRUE_POSITIVE_REWARD", "TRUE_NEGATIVE_REWARD",
                "TRAIN_ATTACK_OVERSAMPLE_RATIO",
                "EPSILON_START", "EPSILON_END", "EPSILON_DECAY",
                "LEARNING_RATE_Q", "BATCH_SIZE", "GAMMA", "TAU",
                "DP_ENABLED", "DP_CLIP_NORM", "DP_NOISE_SCALE",
            ]
        },
    }


def make_serializable(obj):
    if isinstance(obj, (int, float, np.floating)): return float(obj)
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, list): return [make_serializable(i) for i in obj]
    elif isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
    return str(obj)

def save_incremental_checkpoint(filepath, all_histories, all_eval):
    data = {
        "all_histories": make_serializable(all_histories),
        "all_eval": make_serializable(all_eval)
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_incremental_checkpoint(filepath):
    if os.path.exists(filepath):
        print(f"  [+] Loading checkpoint from {filepath}")
        with open(filepath, "r") as f:
            return json.load(f)
    return {"all_histories": {}, "all_eval": {}}


def train_on_real_data(
    algorithm: str, dataset, network: IoTNetwork,
    num_episodes: int = 300, max_steps: int = 100, seed: int = 42,
    use_federated: bool = False, attack_thresholds=None,
):
    """Train a MARL algorithm on real data."""
    device = str(config.DEVICE)
    N = network.num_nodes
    obs_dim = dataset.top_k + 4  # traffic features + node meta

    # Create environment with real data
    env = RealTrafficEnvironment(
        network=network, dataset=dataset,
        max_steps=max_steps, use_test=False, seed=seed,
        attack_thresholds=attack_thresholds,
    )

    # Initialize agents
    if algorithm == "iql":
        agents = [
            IQLAgent(i, state_dim=obs_dim, num_actions=4,
                     device=device, seed=seed+i)
            for i in range(N)
        ]
        is_independent = True
    elif algorithm == "dqn":
        agents = [
            DQNAgent(i, state_dim=obs_dim, num_actions=4, lr=1e-3,
                     device=device, seed=seed+i)
            for i in range(N)
        ]
        is_independent = True
    elif algorithm == "maddpg":
        agents = MADDPGAgent(
            num_agents=N, obs_dim=obs_dim, act_dim=1,
            device=device, seed=seed,
        )
        is_independent = False
    elif algorithm == "qmix":
        global_dim = obs_dim * N + 4
        agents = QMIXAgent(
            num_agents=N, obs_dim=obs_dim, global_state_dim=global_dim,
            num_actions=4, device=device, seed=seed,
        )
        is_independent = False
    
    fed_server = FederatedServer(
        num_agents=N,
        dp_enabled=config.DP_ENABLED,
        dp_clip_norm=config.DP_CLIP_NORM,
        dp_noise_scale=config.DP_NOISE_SCALE,
        energy_safety_threshold=config.ENERGY_SAFETY_THRESHOLD
    ) if (use_federated and is_independent) else None

    history = {
        "episode_rewards": [], "episode_detection_acc": [],
        "episode_fpr": [], "episode_energy": [], "total_comm_bytes": 0,
    }

    total_comm_bytes = 0
    best_acc = 0.0
    pbar = tqdm(range(1, num_episodes + 1), desc=f"{algorithm.upper()} (real data)")

    for episode in pbar:
        obs = env.reset()
        ep_reward = 0.0
        done = False

        if algorithm == "maddpg":
            agents.reset_noise()

        while not done:
            # Select actions
            if algorithm in ("iql", "dqn"):
                actions = np.array([
                    agents[i].select_action(obs[i], explore=True)
                    for i in range(N)
                ])
                cont_actions = None
            elif algorithm == "maddpg":
                cont_actions = agents.select_actions(obs, explore=True)
                actions = agents.discretize_actions(cont_actions)
            elif algorithm == "qmix":
                actions = agents.select_actions(obs, explore=True)
                cont_actions = None

            # Global state BEFORE stepping (must pair with `obs`, not `next_obs`)
            if algorithm == "qmix":
                gs = env.get_global_state()

            # Step
            next_obs, rewards, done, info = env.step(actions)

            # Store
            if algorithm in ("iql", "dqn"):
                for i in range(N):
                    agents[i].store_transition(obs[i], actions[i], rewards[i], next_obs[i], done)
            elif algorithm == "maddpg":
                act_store = cont_actions if cont_actions is not None else actions.astype(np.float32).reshape(-1,1)
                agents.store_transition(obs, act_store, rewards, next_obs, done)
            elif algorithm == "qmix":
                ngs = env.get_global_state()
                agents.store_transition(obs, actions, rewards, next_obs, done, gs, ngs)

            # Update
            if algorithm in ("iql", "dqn"):
                for agent in agents:
                    agent.update()
            else:
                agents.update()

            ep_reward += rewards.mean()
            obs = next_obs
        
        # Federated Averaging Step (adaptive interval & energy-aware selective participation)
        if fed_server is not None and episode > 0:
            # Get current energy levels
            energy_levels = [network.graph.nodes[i]["energy_level"] for i in range(N)]
            mean_energy = np.mean(energy_levels)
            
            # Determine dynamic interval based on remaining network energy
            fed_interval = 10
            for range_name, (threshold, interval) in config.ADAPTIVE_FL_INTERVALS.items():
                if mean_energy >= threshold:
                    fed_interval = interval
                    break
                    
            if episode % 5 == 0 or episode == 1:
                print(f"\n  [Diagnostic Episode {episode}] Mean remaining node energy: {mean_energy:.2f}% | Computed FL Interval: {fed_interval} ep", flush=True)
                
            if episode % fed_interval == 0:
                comm_info = fed_server.aggregate_weights(agents, energy_levels=energy_levels)
                total_comm_bytes += comm_info.get("communication_bytes", 0)
                
                # Print debug information about active/skipped agents and privacy preservation
                active = comm_info.get("active_agents", 0)
                skipped = comm_info.get("skipped_agents", 0)
                priv_score = comm_info.get("privacy_preservation_score", 1.0)
                print(f"\n  [FedAvg Episode {episode}] Dynamic Interval: {fed_interval} ep | "
                      f"Active nodes: {active}/{N} | Skipped (Low Energy): {skipped} | "
                      f"Privacy Score: {priv_score:.2f} | Comm: {comm_info.get('communication_bytes', 0)} bytes", flush=True)

        # Decay exploration
        if algorithm in ("iql", "dqn"):
            for agent in agents:
                agent.decay_epsilon()
        elif algorithm == "qmix":
            agents.decay_epsilon()

        ep_metrics = info.get("episode_metrics", {})
        avg_r = ep_reward / max(1, env.current_step)
        acc = ep_metrics.get("detection_accuracy", 0)
        history["episode_rewards"].append(avg_r)
        history["episode_detection_acc"].append(acc)
        history["episode_fpr"].append(ep_metrics.get("false_positive_rate", 0))
        history["episode_energy"].append(ep_metrics.get("total_energy_consumed", 0))

        if acc > best_acc:
            best_acc = acc
            import os, torch
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/best_{algorithm}_real.pt"
            if algorithm == "maddpg" or algorithm == "qmix":
                agents.save(path)
            else:
                state = {}
                for idx_a, agent in enumerate(agents):
                    state[f"agent_{idx_a}"] = {"q_network": agent.q_network.state_dict()}
                torch.save(state, path)

        if episode % 20 == 0:
            recent_acc = history["episode_detection_acc"][-30:]
            recent_fpr = history["episode_fpr"][-30:]
            pbar.set_postfix({
                "Acc": f"{np.mean(recent_acc):.3f}",
                "FPR": f"{np.mean(recent_fpr):.3f}",
            })

    # Final evaluation on TEST set
    print(f"\n  Evaluating {algorithm.upper()} on TEST set...")
    test_env = RealTrafficEnvironment(
        network=network, dataset=dataset,
        max_steps=max_steps, use_test=True, seed=seed+999,
        attack_thresholds=attack_thresholds,
    )

    eval_metrics = []
    for _ in range(20):
        network.reset()
        obs = test_env.reset()
        done = False
        while not done:
            if algorithm in ("iql", "dqn"):
                actions = np.array([
                    agents[i].select_action(obs[i], explore=False)
                    for i in range(N)
                ])
            elif algorithm == "maddpg":
                cont = agents.select_actions(obs, explore=False)
                actions = agents.discretize_actions(cont)
            elif algorithm == "qmix":
                actions = agents.select_actions(obs, explore=False)
            obs, _, done, info = test_env.step(actions)
        if "episode_metrics" in info:
            eval_metrics.append(info["episode_metrics"])

    eval_result = {
        "mean_reward": np.mean([m["mean_reward"] for m in eval_metrics]),
        "mean_detection_accuracy": np.mean([m["detection_accuracy"] for m in eval_metrics]),
        "mean_fpr": np.mean([m["false_positive_rate"] for m in eval_metrics]),
        "mean_precision": np.mean([m["precision"] for m in eval_metrics]),
        "mean_recall": np.mean([m["recall"] for m in eval_metrics]),
        "mean_f1": np.mean([m["f1_score"] for m in eval_metrics]),
        "mean_mcc": np.mean([m.get("mcc", 0.0) for m in eval_metrics]),
        "mean_energy": np.mean([m["total_energy_consumed"] for m in eval_metrics]),
        "mean_network_lifetime": np.mean([m["network_lifetime_ratio"] for m in eval_metrics]),
        "total_communication_bytes": total_comm_bytes,
    }

    # Convergence speed
    rewards = history["episode_rewards"]
    final_r = np.mean(rewards[-30:]) if rewards else 0
    threshold = 0.95 * final_r if final_r > 0 else 0
    conv_ep = num_episodes
    for i in range(20, len(rewards)):
        if np.mean(rewards[max(0,i-20):i]) >= threshold:
            conv_ep = i; break
    history["convergence_episode"] = conv_ep

    # Generate XAI SHAP plots for DQN to explain decisions
    if algorithm == "dqn":
        explain_real_dataset_agent(agents[0], test_env, num_samples=100)

    return history, eval_result


def _load_real_dataset(dataset_name):
    if dataset_name == "unsw-nb15":
        dataset = UNSWNB15Processor(top_k=15, seed=42)
    elif dataset_name == "cicids2017":
        dataset = CICIDS2017Processor(top_k=15, seed=42)
    else:
        dataset = NSLKDDProcessor(top_k=15, seed=42)

    dataset.load()
    stats = dataset.get_stats()
    print(f"\n  Dataset stats:")
    print(f"    Train: {stats['train_samples']} samples")
    print(f"    Test:  {stats['test_samples']} samples")
    print(f"    Attack ratio (train): {stats['attack_ratio_train']:.2%}")
    return dataset


def _run_all_methods_for_seed(dataset, dataset_name, seed=42):
    """
    Train/evaluate all MARL algorithms, FL variants, and baselines for ONE seed.

    Reuses the legacy (non-suffixed) checkpoint for seed=42 so previously
    computed single-seed results aren't recomputed; other seeds get their
    own checkpoint file so a multi-seed run can resume if interrupted.
    """
    network = IoTNetwork(num_nodes=20, topology="random", seed=seed)
    game = create_game_from_network(network, seed=seed)

    suffix = "" if seed == 42 else f"_seed{seed}"
    checkpoint_file = os.path.join(config.RESULTS_DIR, f"checkpoint_real_{dataset_name}{suffix}.json")
    checkpoint_data = load_incremental_checkpoint(checkpoint_file)

    all_histories = checkpoint_data.get("all_histories", {})
    all_eval = checkpoint_data.get("all_eval", {})

    # ── Train all MARL algorithms ──
    for algo in ["iql", "dqn", "maddpg", "qmix"]:
        if algo not in all_eval:
            print(f"\n{'='*50}")
            print(f"  Training {algo.upper()} on {dataset_name.upper()} (seed={seed})")
            print(f"{'='*50}")
            network.reset()
            h, e = train_on_real_data(algo, dataset, network, num_episodes=300, seed=seed,
                                       attack_thresholds=game.attack_thresholds)
            all_histories[algo] = h
            all_eval[algo] = e
            print(f"  {algo.upper()} TEST: Acc={e['mean_detection_accuracy']:.4f}, "
                  f"FPR={e['mean_fpr']:.4f}, F1={e['mean_f1']:.4f}")
            save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
        else:
            print(f"\n  >>> Skipping {algo.upper()} (loaded from checkpoint, seed={seed})")

    # ── Federated Learning Ablation (IQL and DQN with FedAvg) ──
    for algo in ["iql", "dqn"]:
        key = f"{algo}_federated"
        if key not in all_eval:
            print(f"\n{'='*50}")
            print(f"  Training {algo.upper()} + FedAvg on {dataset_name.upper()} (seed={seed})")
            print(f"{'='*50}")
            network.reset()
            h, e = train_on_real_data(algo, dataset, network, num_episodes=300, seed=seed, use_federated=True,
                                       attack_thresholds=game.attack_thresholds)
            all_histories[key] = h
            all_eval[key] = e
            print(f"  {algo.upper()}+FL TEST: Acc={e['mean_detection_accuracy']:.4f}, "
                  f"FPR={e['mean_fpr']:.4f}, F1={e['mean_f1']:.4f}, "
                  f"Comm={e['total_communication_bytes']:.0f} bytes")
            save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
        else:
            print(f"\n  >>> Skipping {key.upper()} (loaded from checkpoint, seed={seed})")

    # ── Baselines on real data ──
    print(f"\n{'='*50}")
    print(f"  Running baselines on {dataset_name.upper()} (seed={seed})")
    print(f"{'='*50}")

    # Random
    if "random" not in all_eval:
        network.reset()
        rp = RandomPolicy(20, seed=seed)
        test_env_r = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["random"] = evaluate_baseline(rp, test_env_r, 20, "Random")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping Random Baseline (loaded from checkpoint)")

    # Always Monitor
    if "always_monitor" not in all_eval:
        network.reset()
        am = AlwaysMonitorPolicy(20)
        test_env_a = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["always_monitor"] = evaluate_baseline(am, test_env_a, 20, "Always Monitor")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping Always Monitor Baseline (loaded from checkpoint)")

    # Static Stackelberg
    if "static_stackelberg" not in all_eval:
        network.reset()
        sp = StaticStackelbergPolicy(game)
        test_env_s = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["static_stackelberg"] = evaluate_baseline(sp, test_env_s, 20, "Static Stackelberg")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping Static Stackelberg Baseline (loaded from checkpoint)")

    # Baseline 3.5: Adaptive Stackelberg
    if "adaptive_stackelberg" not in all_eval:
        network.reset()
        adaptive_game = AdaptiveStackelbergGame(
            num_nodes=game.N,
            detection_rewards=game.R,
            monitoring_costs=game.C,
            damage_values=game.D,
            detection_penalties=game.P,
            defender_budget=game.B_def,
            attacker_budget=game.B_att
        )
        asp = AdaptiveStackelbergPolicy(adaptive_game)
        test_env_a = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["adaptive_stackelberg"] = evaluate_baseline(asp, test_env_a, 20, "Adaptive Stackelberg")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping Adaptive Stackelberg Baseline (loaded from checkpoint)")

    # Deep Baseline: LSTM
    if "lstm_ids" not in all_eval:
        network.reset()
        lstm = LSTM_Baseline(num_nodes=20, input_dim=dataset.top_k, seed=seed, device=str(config.DEVICE))
        real_data_dict = {"X_train": dataset.X_train, "y_train": dataset.y_train, "X_test": dataset.X_test, "y_test": dataset.y_test}
        lstm.train_classifier(dataset_override=real_data_dict)
        test_env_l = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["lstm_ids"] = evaluate_baseline(lstm, test_env_l, 20, "LSTM IDS")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping LSTM IDS Baseline (loaded from checkpoint)")

    # Deep Baseline: Autoencoder
    if "autoencoder" not in all_eval:
        network.reset()
        ae = Autoencoder_Baseline(num_nodes=20, input_dim=dataset.top_k, seed=seed, device=str(config.DEVICE))
        real_data_dict = {"X_train": dataset.X_train, "y_train": dataset.y_train, "X_test": dataset.X_test, "y_test": dataset.y_test}
        ae.train_classifier(dataset_override=real_data_dict)
        test_env_ae = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["autoencoder"] = evaluate_baseline(ae, test_env_ae, 20, "Autoencoder")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping Autoencoder Baseline (loaded from checkpoint)")

    # Deep Baseline: CNN
    if "cnn_ids" not in all_eval:
        network.reset()
        cnn = CNN_Baseline(num_nodes=20, input_dim=dataset.top_k, seed=seed, device=str(config.DEVICE))
        real_data_dict = {"X_train": dataset.X_train, "y_train": dataset.y_train, "X_test": dataset.X_test, "y_test": dataset.y_test}
        cnn.train_classifier(dataset_override=real_data_dict)
        test_env_cnn = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["cnn_ids"] = evaluate_baseline(cnn, test_env_cnn, 20, "CNN IDS")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping CNN IDS Baseline (loaded from checkpoint)")

    # Deep Baseline: Transformer
    if "transformer_ids" not in all_eval:
        network.reset()
        transformer = Transformer_Baseline(num_nodes=20, input_dim=dataset.top_k, seed=seed, device=str(config.DEVICE))
        real_data_dict = {"X_train": dataset.X_train, "y_train": dataset.y_train, "X_test": dataset.X_test, "y_test": dataset.y_test}
        transformer.train_classifier(dataset_override=real_data_dict)
        test_env_tr = RealTrafficEnvironment(network=network, dataset=dataset, max_steps=100, use_test=True, seed=seed+999, attack_thresholds=game.attack_thresholds)
        all_eval["transformer_ids"] = evaluate_baseline(transformer, test_env_tr, 20, "Transformer IDS")
        save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)
    else:
        print("\n  >>> Skipping Transformer IDS Baseline (loaded from checkpoint)")

    return all_histories, all_eval


def run_real_dataset_experiments(dataset_name="nsl-kdd", seed=42):
    """Run complete experiments on real dataset for a single seed (original pipeline)."""
    print("\n" + "#" * 60)
    print(f"  REAL DATASET EXPERIMENTS ({dataset_name.upper()}, seed={seed})")
    print("#" * 60)

    dataset = _load_real_dataset(dataset_name)
    all_histories, all_eval = _run_all_methods_for_seed(dataset, dataset_name, seed=seed)

    # ── Generate outputs ──
    out_dir = os.path.join(config.FIGURES_DIR, "real_dataset")
    os.makedirs(out_dir, exist_ok=True)
    table_dir = os.path.join(config.TABLES_DIR)

    # Plots
    plot_training_curves(all_histories, output_dir=out_dir)
    plot_training_curves(all_histories, metric="episode_detection_acc",
                        ylabel="Detection Accuracy", title=f"Detection Accuracy ({dataset_name.upper()})",
                        filename=f"real_{dataset_name}_detection_accuracy.png", output_dir=out_dir)
    plot_fpr_comparison(all_eval, filename=f"real_{dataset_name}_fpr_comparison.png", output_dir=out_dir)
    plot_confusion_matrix(all_eval, filename=f"real_{dataset_name}_confusion_matrix.png", output_dir=out_dir)
    plot_convergence_comparison(all_histories, filename=f"real_{dataset_name}_convergence.png", output_dir=out_dir)

    # Comprehensive comparison bar chart
    _plot_real_comparison(all_eval, out_dir, dataset_name)

    # Table
    import pandas as pd
    rows = []
    for method, r in all_eval.items():
        rows.append({
            "Method": method.upper(),
            "Detection Acc.": f"{r['mean_detection_accuracy']:.4f}",
            "FPR": f"{r['mean_fpr']:.4f}",
            "Precision": f"{r.get('mean_precision',0):.4f}",
            "Recall": f"{r.get('mean_recall',0):.4f}",
            "F1": f"{r.get('mean_f1',0):.4f}",
            "Energy": f"{r['mean_energy']:.1f}",
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(table_dir, f"real_dataset_{dataset_name}_comparison.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\n  Table saved: {csv_path}")
    print(f"  Figures saved: {out_dir}")

    # Save JSON
    results_json = {
        method: {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                 for k, v in r.items()}
        for method, r in all_eval.items()
    }
    json_path = os.path.join(config.RESULTS_DIR, f"real_dataset_{dataset_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  JSON saved: {json_path}")

    print(f"\n{'#'*60}")
    print("  REAL DATASET EXPERIMENTS COMPLETE!")
    print(f"{'#'*60}\n")


METRIC_KEYS = [
    "mean_reward", "mean_detection_accuracy", "mean_fpr",
    "mean_precision", "mean_recall", "mean_f1", "mean_mcc", "mean_energy",
    "mean_network_lifetime",
]


def _aggregate_seeds(per_seed_eval):
    """
    per_seed_eval: {seed: {method: {metric: value}}}
    Returns {method: {metric: {"values": [...], "mean": x, "std": x, "n": k}}}
    """
    methods = set()
    for seed_results in per_seed_eval.values():
        methods.update(seed_results.keys())

    aggregated = {}
    for method in methods:
        aggregated[method] = {}
        for metric in METRIC_KEYS:
            values = [
                seed_results[method][metric]
                for seed_results in per_seed_eval.values()
                if method in seed_results and metric in seed_results[method]
            ]
            if not values:
                continue
            aggregated[method][metric] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return aggregated


def _paired_significance_tests(per_seed_eval, metric="mean_detection_accuracy", alpha=0.05):
    """
    Paired t-test between every pair of methods that have results on the SAME
    set of seeds, on a single metric. With only a handful of seeds this has
    limited statistical power -- treat p-values as indicative, not definitive.
    """
    seeds_sorted = sorted(per_seed_eval.keys())
    methods = set()
    for seed_results in per_seed_eval.values():
        methods.update(seed_results.keys())
    methods = sorted(methods)

    rows = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            v1, v2 = [], []
            for s in seeds_sorted:
                sr = per_seed_eval[s]
                if m1 in sr and m2 in sr and metric in sr[m1] and metric in sr[m2]:
                    v1.append(sr[m1][metric])
                    v2.append(sr[m2][metric])
            if len(v1) < 2:
                continue
            mean_diff = float(np.mean(v1) - np.mean(v2))
            try:
                t_stat, p_value = scipy_stats.ttest_rel(v1, v2)
                p_value = float(p_value)
            except Exception:
                t_stat, p_value = float("nan"), float("nan")
            rows.append({
                "method_a": m1, "method_b": m2, "metric": metric,
                "n_seeds": len(v1), "mean_diff_a_minus_b": mean_diff,
                "t_stat": float(t_stat) if t_stat == t_stat else None,
                "p_value": p_value if p_value == p_value else None,
                "significant_at_0.05": bool(p_value is not None and p_value == p_value and p_value < alpha),
            })
    return rows


def run_multiseed_experiments(dataset_name="nsl-kdd", seeds=None):
    """
    Run the full method suite across multiple seeds and report mean +/- std
    plus paired significance tests, instead of a single-run point estimate.
    Per-seed checkpoints allow an interrupted run to resume where it left off.
    """
    if seeds is None:
        seeds = tuple(config.SEEDS)
    print("\n" + "#" * 60)
    print(f"  MULTI-SEED REAL DATASET EXPERIMENTS ({dataset_name.upper()})")
    print(f"  Seeds: {list(seeds)}")
    print("#" * 60)

    dataset = _load_real_dataset(dataset_name)

    per_seed_eval = {}
    for seed in seeds:
        print(f"\n{'#'*50}\n  SEED {seed}\n{'#'*50}")
        _, all_eval = _run_all_methods_for_seed(dataset, dataset_name, seed=seed)
        per_seed_eval[seed] = all_eval

    aggregated = _aggregate_seeds(per_seed_eval)
    sig_acc = _paired_significance_tests(per_seed_eval, metric="mean_detection_accuracy")
    sig_f1 = _paired_significance_tests(per_seed_eval, metric="mean_f1")

    # ── Save aggregated results JSON ──
    agg_path = os.path.join(config.RESULTS_DIR, f"real_dataset_{dataset_name}_multiseed_results.json")
    with open(agg_path, "w") as f:
        json.dump(make_serializable({
            "run_metadata": collect_run_metadata(seeds),
            "seeds": list(seeds),
            "per_seed": per_seed_eval,
            "aggregated": aggregated,
        }), f, indent=2)
    print(f"\n  Aggregated JSON saved: {agg_path}")

    # ── Save significance test results ──
    sig_path = os.path.join(config.RESULTS_DIR, f"real_dataset_{dataset_name}_significance.json")
    with open(sig_path, "w") as f:
        json.dump({
            "note": (
                f"Paired t-tests across n={len(seeds)} seeds. With this few seeds, "
                "p-values are indicative rather than definitive -- treat as a "
                "supplementary signal alongside the mean/std comparison, not a "
                "sole basis for claims of significance."
            ),
            "detection_accuracy": sig_acc,
            "f1_score": sig_f1,
        }, f, indent=2)
    print(f"  Significance tests saved: {sig_path}")

    # ── Comparison table with mean +/- std ──
    import pandas as pd
    rows = []
    marl_order = ["iql", "dqn", "maddpg", "qmix", "iql_federated", "dqn_federated"]
    baseline_order = ["random", "always_monitor", "static_stackelberg", "adaptive_stackelberg",
                       "lstm_ids", "autoencoder", "cnn_ids", "transformer_ids"]
    method_order = [m for m in marl_order + baseline_order if m in aggregated]
    method_order += [m for m in aggregated if m not in method_order]

    for method in method_order:
        m = aggregated[method]
        def fmt(key, pct=False):
            if key not in m:
                return "-"
            mean, std = m[key]["mean"], m[key]["std"]
            if pct:
                return f"{mean*100:.2f} +/- {std*100:.2f}"
            return f"{mean:.2f} +/- {std:.2f}"
        rows.append({
            "Method": method.upper(),
            "Detection Acc. (%)": fmt("mean_detection_accuracy", pct=True),
            "FPR (%)": fmt("mean_fpr", pct=True),
            "Precision (%)": fmt("mean_precision", pct=True),
            "Recall (%)": fmt("mean_recall", pct=True),
            "F1 (%)": fmt("mean_f1", pct=True),
            "MCC": fmt("mean_mcc"),
            "Energy (J)": fmt("mean_energy"),
            "N seeds": aggregated[method].get("mean_detection_accuracy", {}).get("n", 0),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.TABLES_DIR, f"real_dataset_{dataset_name}_comparison_multiseed.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} MULTI-SEED RESULTS (mean +/- std, n={len(seeds)})")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\n  Table saved: {csv_path}")

    # ── Error-bar comparison plot ──
    _plot_multiseed_comparison(aggregated, method_order, dataset_name)

    print(f"\n{'#'*60}")
    print("  MULTI-SEED EXPERIMENTS COMPLETE!")
    print(f"{'#'*60}\n")

    return aggregated, sig_acc, sig_f1


def _plot_multiseed_comparison(aggregated, method_order, dataset_name):
    setup_style()
    out_dir = os.path.join(config.FIGURES_DIR, "real_dataset")
    os.makedirs(out_dir, exist_ok=True)

    methods = [m for m in method_order if m in aggregated and "mean_detection_accuracy" in aggregated[m]]
    display = [m.upper().replace("_", " ") for m in methods]
    means = [aggregated[m]["mean_detection_accuracy"]["mean"] * 100 for m in methods]
    stds = [aggregated[m]["mean_detection_accuracy"]["std"] * 100 for m in methods]
    colors = [config.COLORS.get(m, "#AAAAAA") for m in methods]

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.1), 6))
    bars = ax.bar(display, means, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Detection Accuracy (%)", fontsize=11)
    ax.set_title(f"Detection Accuracy across seeds, mean ± std ({dataset_name.upper()})",
                 fontweight="bold", fontsize=14)
    ax.set_xticks(range(len(display)))
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"real_{dataset_name}_multiseed_accuracy.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved: {fname}")


def _plot_real_comparison(eval_results, output_dir, dataset_name="nsl-kdd"):
    """Plot comprehensive comparison on real data."""
    setup_style()

    marl = ["iql", "dqn", "maddpg", "qmix"]
    baselines = ["random", "always_monitor", "static_stackelberg", "adaptive_stackelberg", "lstm_ids", "autoencoder", "cnn_ids", "transformer_ids"]

    methods = [m for m in marl + baselines if m in eval_results]
    display = [m.upper().replace("_", " ") for m in methods]

    # Convert fractions to percentages for readability
    accs = [eval_results[m]["mean_detection_accuracy"] * 100 for m in methods]
    fprs = [eval_results[m]["mean_fpr"] * 100 for m in methods]
    f1s = [eval_results[m].get("mean_f1", 0) * 100 for m in methods]
    energies = [eval_results[m]["mean_energy"] for m in methods]

    colors = [config.COLORS.get(m, "#AAAAAA") for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_configs = [
        (axes[0,0], accs, "Detection Accuracy", "Accuracy (%)", "{:.2f}%"),
        (axes[0,1], fprs, "False Positive Rate", "FPR (%)", "{:.2f}%"),
        (axes[1,0], f1s, "F1 Score", "F1 Score (%)", "{:.2f}%"),
        (axes[1,1], energies, "Energy Consumption", "Energy (Joules)", "{:.0f} J"),
    ]

    for ax, vals, title, ylabel, fmt_str in plot_configs:
        bars = ax.bar(display, vals, color=colors, edgecolor="white", linewidth=1.5)
        
        max_val = max(vals) if vals else 1.0
        if max_val == 0:
            max_val = 1.0
            
        # Give 30% headroom for text labels
        ax.set_ylim(0, max_val * 1.3)
        
        for b, v in zip(bars, vals):
            label_text = fmt_str.format(v)
            # Position text offset by 2% of max value above the bar
            ax.text(
                b.get_x() + b.get_width()/2, 
                b.get_height() + 0.02 * max_val,
                label_text, 
                ha="center", 
                va="bottom", 
                fontsize=8, 
                fontweight="bold", 
                rotation=90
            )
            
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(range(len(display)))
        ax.set_xticklabels(display, rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"MARL vs Baselines on {dataset_name.upper()} Dataset",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"real_{dataset_name}_full_comparison.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved: real_{dataset_name}_full_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["iql", "dqn", "maddpg", "qmix"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--dataset", type=str, default="nsl-kdd",
                        choices=["nsl-kdd", "unsw-nb15", "cicids2017"])
    parser.add_argument("--federated", action="store_true", help="Enable Privacy-Preserving Federated Learning for independent agents")
    parser.add_argument("--multiseed", action="store_true",
                         help="Run the full method suite across multiple seeds and report mean +/- std with paired significance tests")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Seeds to use with --multiseed (default: all of config.SEEDS)")
    args = parser.parse_args()

    if args.multiseed:
        seeds = args.seeds if args.seeds else config.SEEDS
        run_multiseed_experiments(args.dataset, seeds=tuple(seeds))
    elif args.algorithm:
        # Single algorithm
        if args.dataset == "unsw-nb15":
            dataset = UNSWNB15Processor(top_k=15, seed=42).load()
        elif args.dataset == "cicids2017":
            dataset = CICIDS2017Processor(top_k=15, seed=42).load()
        else:
            dataset = NSLKDDProcessor(top_k=15, seed=42).load()
            
        network = IoTNetwork(num_nodes=20, seed=42)
        game = create_game_from_network(network, seed=42)
        h, e = train_on_real_data(args.algorithm, dataset, network,
                                  num_episodes=args.episodes,
                                  use_federated=args.federated,
                                  attack_thresholds=game.attack_thresholds)
        prefix = "FEDERATED " if args.federated else ""
        print(f"\n  {prefix}{args.algorithm.upper()} on {args.dataset.upper()}:")
        for k, v in e.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    else:
        run_real_dataset_experiments(args.dataset)
