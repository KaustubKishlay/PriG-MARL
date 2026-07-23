"""
Batch Experiment Runner for Paper.

Runs all experiments needed for IEEE paper:
1. Compare: IQL vs DQN vs MADDPG vs QMIX
2. Compare: With vs without Stackelberg attacker
3. Ablation: Varying network sizes (10, 20, 50)
4. Ablation: Varying energy budgets

Generates all tables and figures.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from training.trainer import MARLTrainer
from training.evaluator import Evaluator
from visualization.plots import generate_all_plots
from visualization.dashboard import generate_dashboard


def run_algorithm_comparison(
    algorithms=None, num_episodes=500, network_size=20, seeds=None
):
    """Experiment 1: Compare all MARL algorithms."""
    algorithms = algorithms or ["iql", "dqn", "maddpg", "qmix"]
    seeds = seeds or [42]

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Algorithm Comparison")
    print("=" * 60)

    evaluator = Evaluator()
    all_histories = {}
    all_eval_results = {}
    last_network = None
    last_eq = None

    for algo in algorithms:
        for seed in seeds:
            print(f"\n  >>> {algo.upper()} (seed={seed})")
            trainer = MARLTrainer(
                algorithm=algo,
                num_nodes=network_size,
                num_episodes=num_episodes,
                attacker_strategy="strategic",
                seed=seed,
            )
            last_network = trainer.network
            last_eq = trainer.get_equilibrium_info()

            history = trainer.train()
            evaluator.add_result(algo, seed, history)

            if algo not in all_histories:
                all_histories[algo] = history
            final_eval = trainer.evaluate(num_episodes=20)
            all_eval_results[algo] = final_eval

    # Print and save results
    evaluator.print_summary()
    evaluator.save_tables()

    # Generate all plots
    generate_all_plots(all_histories, all_eval_results, last_network, last_eq)
    generate_dashboard(all_histories, all_eval_results, last_eq)

    return evaluator, all_histories, all_eval_results


def run_attacker_comparison(algorithm="dqn", num_episodes=500, network_size=20):
    """Experiment 2: Random vs Strategic vs Stackelberg attacker."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Attacker Strategy Comparison")
    print("=" * 60)

    strategies = ["random", "strategic", "stackelberg"]
    histories = {}
    eval_results = {}

    for strategy in strategies:
        print(f"\n  >>> {algorithm.upper()} vs {strategy} attacker")
        trainer = MARLTrainer(
            algorithm=algorithm,
            num_nodes=network_size,
            num_episodes=num_episodes,
            attacker_strategy=strategy,
            seed=42,
        )
        history = trainer.train()
        final_eval = trainer.evaluate(num_episodes=20)

        key = f"{algorithm}_{strategy}"
        histories[key] = history
        eval_results[key] = final_eval

    # Save comparison
    output_dir = os.path.join(config.FIGURES_DIR, "attacker_comparison")
    generate_all_plots(histories, eval_results, output_dir=output_dir)

    return histories, eval_results


def run_network_size_ablation(algorithm="dqn", num_episodes=300, sizes=None):
    """Experiment 3: Varying network sizes."""
    sizes = sizes or [10, 20]
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Network Size Ablation")
    print("=" * 60)

    histories = {}
    eval_results = {}

    for N in sizes:
        print(f"\n  >>> {algorithm.upper()} with N={N} nodes")
        trainer = MARLTrainer(
            algorithm=algorithm,
            num_nodes=N,
            num_episodes=num_episodes,
            seed=42,
        )
        history = trainer.train()
        final_eval = trainer.evaluate(num_episodes=10)

        key = f"{algorithm}_N{N}"
        histories[key] = history
        eval_results[key] = final_eval

    output_dir = os.path.join(config.FIGURES_DIR, "size_ablation")
    generate_all_plots(histories, eval_results, output_dir=output_dir)

    return histories, eval_results


def run_all_experiments():
    """Run all experiments for the paper."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'#'*60}")
    print(f"  FULL EXPERIMENT SUITE — {timestamp}")
    print(f"{'#'*60}")

    results = {}

    # Experiment 1: Algorithm comparison (reduced for speed)
    ev, h1, e1 = run_algorithm_comparison(
        algorithms=["iql", "dqn", "maddpg", "qmix"],
        num_episodes=500,
        network_size=20,
    )
    results["algorithm_comparison"] = {"histories": {k: {"convergence": v.get("convergence_episode",0)} for k,v in h1.items()},
                                       "eval": {k: {kk: float(vv) for kk,vv in v.items()} for k,v in e1.items()}}

    # Experiment 2: Attacker comparison
    h2, e2 = run_attacker_comparison(algorithm="dqn", num_episodes=300)
    results["attacker_comparison"] = {"eval": {k: {kk: float(vv) for kk,vv in v.items()} for k,v in e2.items()}}

    # Experiment 3: Network size ablation
    h3, e3 = run_network_size_ablation(algorithm="dqn", num_episodes=200, sizes=[10, 20])
    results["size_ablation"] = {"eval": {k: {kk: float(vv) for kk,vv in v.items()} for k,v in e3.items()}}

    # Save all results
    results_path = os.path.join(config.RESULTS_DIR, f"all_experiments_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  All results saved to {results_path}")

    print(f"\n{'#'*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "compare", "attacker", "ablation"],
                        help="Which experiment to run")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--network_size", type=int, default=20)
    args = parser.parse_args()

    if args.experiment == "all":
        run_all_experiments()
    elif args.experiment == "compare":
        run_algorithm_comparison(num_episodes=args.episodes, network_size=args.network_size)
    elif args.experiment == "attacker":
        run_attacker_comparison(num_episodes=args.episodes, network_size=args.network_size)
    elif args.experiment == "ablation":
        run_network_size_ablation(num_episodes=args.episodes)
