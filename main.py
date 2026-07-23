"""
Main entry point for Game-Theoretic IDS with MARL.

Usage:
    python main.py --algorithm maddpg --episodes 500 --network_size 20
    python main.py --algorithm dqn --attacker strategic --topology scale_free
    python main.py --algorithm qmix --episodes 1000 --seed 42
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from training.trainer import MARLTrainer
from training.evaluator import Evaluator
from visualization.plots import generate_all_plots
from visualization.dashboard import generate_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="Game-Theoretic IDS with Multi-Agent RL"
    )
    parser.add_argument("--algorithm", type=str, default="dqn",
                        choices=["iql", "dqn", "maddpg", "qmix"],
                        help="MARL algorithm to use")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes")
    parser.add_argument("--network_size", type=int, default=None,
                        help="Number of IoT nodes")
    parser.add_argument("--topology", type=str, default=None,
                        choices=["random", "scale_free", "grid", "star"],
                        help="Network topology type")
    parser.add_argument("--attacker", type=str, default="strategic",
                        choices=["random", "strategic", "stackelberg"],
                        help="Attacker strategy type")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--no_dashboard", action="store_true",
                        help="Skip dashboard generation")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Game-Theoretic Intrusion Detection System")
    print("  Multi-Agent Reinforcement Learning for IoT Networks")
    print("=" * 60)

    # Override config with CLI args
    if args.episodes:
        config.NUM_EPISODES = args.episodes
    if args.network_size:
        config.NETWORK_SIZE = args.network_size
    if args.max_steps:
        config.MAX_STEPS_PER_EPISODE = args.max_steps

    # Initialize trainer
    trainer = MARLTrainer(
        algorithm=args.algorithm,
        num_nodes=args.network_size or config.NETWORK_SIZE,
        topology=args.topology or config.NETWORK_TOPOLOGY,
        num_episodes=args.episodes or config.NUM_EPISODES,
        max_steps=args.max_steps or config.MAX_STEPS_PER_EPISODE,
        attacker_strategy=args.attacker,
        seed=args.seed,
        device=args.device,
    )

    # Print Stackelberg equilibrium info
    eq = trainer.get_equilibrium_info()
    print(f"\n  Stackelberg Equilibrium:")
    print(f"    Defender Utility: {eq['defender_utility']:.4f}")
    print(f"    Attacker Utility: {eq['attacker_utility']:.4f}")
    print(f"    Convergence: {eq['convergence']}")

    # Train
    history = trainer.train()

    # Final evaluation
    print("\n  Running final evaluation (20 episodes)...")
    final_eval = trainer.evaluate(num_episodes=20)
    print(f"\n  Final Results:")
    print(f"    Detection Accuracy: {final_eval['mean_detection_accuracy']:.4f}")
    print(f"    False Positive Rate: {final_eval['mean_fpr']:.4f}")
    print(f"    Precision:          {final_eval['mean_precision']:.4f}")
    print(f"    Recall:             {final_eval['mean_recall']:.4f}")
    print(f"    F1 Score:           {final_eval['mean_f1']:.4f}")
    print(f"    Energy Consumed:    {final_eval['mean_energy']:.1f}")
    print(f"    Network Lifetime:   {final_eval['mean_network_lifetime']:.4f}")

    # Save results
    results_path = os.path.join(config.RESULTS_DIR, f"{args.algorithm}_results.json")
    save_data = {
        "algorithm": args.algorithm,
        "seed": args.seed,
        "num_nodes": args.network_size or config.NETWORK_SIZE,
        "topology": args.topology or config.NETWORK_TOPOLOGY,
        "episodes": args.episodes or config.NUM_EPISODES,
        "final_eval": {k: float(v) for k, v in final_eval.items()},
        "convergence_episode": history.get("convergence_episode", 0),
        "equilibrium": {
            "defender_utility": float(eq["defender_utility"]),
            "attacker_utility": float(eq["attacker_utility"]),
        },
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Generate plots
    if not args.no_plots:
        histories = {args.algorithm: history}
        eval_results = {args.algorithm: final_eval}
        generate_all_plots(
            histories, eval_results,
            network=trainer.network,
            equilibrium=eq,
        )

    # Generate dashboard
    if not args.no_dashboard:
        histories = {args.algorithm: history}
        eval_results = {args.algorithm: final_eval}
        generate_dashboard(histories, eval_results, eq)

    print("\n  Done!\n")


if __name__ == "__main__":
    main()
