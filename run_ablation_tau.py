"""
Reward ablation: does the Stackelberg deterrence threshold tau_i actually
matter, or would any generic energy penalty do the same job?

Trains the SAME agent (DQN by default) on the SAME real dataset under three
reward configurations that differ ONLY in the deterrence term of Eq. (8):

  stackelberg  per-node tau_i = D_i / (D_i + P_i) from the Stackelberg game
               (the paper's proposed method, unchanged)
  constant     a flat threshold equal to mean(tau_i) on every node -- same
               average shaping pressure, but no game-theoretic per-node
               allocation
  none         deterrence term removed entirely; only the base energy
               penalty of Eq. (8) remains

Multi-seed, checkpointed, resumable -- mirrors run_real_dataset.py
conventions. Reports mean +/- std per mode plus paired t-tests between
modes on accuracy, energy, and network lifetime.

Usage (from project root):
  python -u run_ablation_tau.py --dataset cicids2017
  python -u run_ablation_tau.py --dataset nsl-kdd --seeds 42 123
  python -u run_ablation_tau.py --dataset cicids2017 --algorithm dqn --episodes 300
"""

import os, json, argparse
import numpy as np
from scipy import stats as scipy_stats

import config
from network.iot_network import IoTNetwork
from game.stackelberg import create_game_from_network
from run_real_dataset import (
    train_on_real_data, _load_real_dataset,
    save_incremental_checkpoint, load_incremental_checkpoint,
    make_serializable, collect_run_metadata,
)

MODES = ["stackelberg", "constant", "none"]

# Metrics reported per mode and significance-tested between modes.
METRICS = [
    "mean_detection_accuracy", "mean_fpr", "mean_precision", "mean_recall",
    "mean_f1", "mean_energy", "mean_network_lifetime",
]


def thresholds_for_mode(mode, game, num_nodes):
    if mode == "stackelberg":
        return np.asarray(game.attack_thresholds)
    if mode == "constant":
        return np.full(num_nodes, float(np.mean(game.attack_thresholds)))
    return None  # "none"


def run_ablation(dataset_name, seeds, algorithm="dqn", episodes=300):
    dataset = _load_real_dataset(dataset_name)

    checkpoint_file = os.path.join(
        config.RESULTS_DIR, f"checkpoint_ablation_tau_{dataset_name}.json")
    ckpt = load_incremental_checkpoint(checkpoint_file)
    all_eval = ckpt.get("all_eval", {})       # keyed "<mode>_seed<seed>"
    all_histories = ckpt.get("all_histories", {})

    for seed in seeds:
        for mode in MODES:
            key = f"{mode}_seed{seed}"
            if key in all_eval:
                print(f"\n  >>> Skipping {key} (loaded from checkpoint)")
                continue
            print(f"\n{'='*50}")
            print(f"  Ablation [{mode}] {algorithm.upper()} on "
                  f"{dataset_name.upper()} (seed={seed})")
            print(f"{'='*50}")
            network = IoTNetwork(num_nodes=20, topology="random", seed=seed)
            game = create_game_from_network(network, seed=seed)
            network.reset()
            h, e = train_on_real_data(
                algorithm, dataset, network,
                num_episodes=episodes, seed=seed,
                attack_thresholds=thresholds_for_mode(mode, game, network.num_nodes),
            )
            all_histories[key] = h
            all_eval[key] = e
            print(f"  [{mode}] TEST: Acc={e['mean_detection_accuracy']:.4f}, "
                  f"Energy={e['mean_energy']:.1f} J, "
                  f"Lifetime={e['mean_network_lifetime']:.2%}")
            save_incremental_checkpoint(checkpoint_file, all_histories, all_eval)

    # ── Aggregate: mode -> metric -> per-seed values ──
    per_mode = {m: {k: [] for k in METRICS} for m in MODES}
    for seed in seeds:
        for mode in MODES:
            e = all_eval.get(f"{mode}_seed{seed}")
            if e is None:
                continue
            for k in METRICS:
                per_mode[mode][k].append(float(e[k]))

    summary = {
        mode: {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for k, v in metrics.items() if v
        }
        for mode, metrics in per_mode.items()
    }

    # ── Paired t-tests between modes, per metric ──
    significance = {}
    pairs = [("stackelberg", "constant"), ("stackelberg", "none"),
             ("constant", "none")]
    for a, b in pairs:
        for k in METRICS:
            va, vb = per_mode[a][k], per_mode[b][k]
            if len(va) >= 2 and len(va) == len(vb):
                diffs = np.array(va) - np.array(vb)
                if np.allclose(diffs, 0):
                    t, p = 0.0, 1.0
                else:
                    t, p = scipy_stats.ttest_rel(va, vb)
                significance[f"{a}_vs_{b}::{k}"] = {
                    "t": float(t), "p": float(p), "n_seeds": len(va),
                    "mean_diff_a_minus_b": float(np.mean(diffs)),
                }

    out = {
        "run_metadata": collect_run_metadata(seeds),
        "note": ("Paired t-tests across seeds; with few seeds this has low "
                 "power, so interpret non-significance cautiously."),
        "dataset": dataset_name, "algorithm": algorithm,
        "episodes": episodes, "seeds": list(seeds),
        "summary": summary, "significance": significance,
    }
    out_file = os.path.join(config.RESULTS_DIR,
                            f"ablation_tau_{dataset_name}_results.json")
    with open(out_file, "w") as f:
        json.dump(make_serializable(out), f, indent=2)
    print(f"\n  Ablation results saved: {out_file}")

    # ── Console + CSV table ──
    os.makedirs(os.path.join(config.RESULTS_DIR, "tables"), exist_ok=True)
    csv_file = os.path.join(config.RESULTS_DIR, "tables",
                            f"ablation_tau_{dataset_name}.csv")
    header = ["Reward mode"] + METRICS + ["N seeds"]
    lines = [",".join(header)]
    print(f"\n{'='*60}")
    print(f"  TAU-ABLATION RESULTS ({dataset_name.upper()}, {algorithm.upper()})")
    print(f"{'='*60}")
    for mode in MODES:
        s = summary.get(mode, {})
        if not s:
            continue
        row = [mode]
        for k in METRICS:
            row.append(f"{s[k]['mean']:.4f} +/- {s[k]['std']:.4f}")
        row.append(str(s[METRICS[0]]["n"]))
        lines.append(",".join(row))
        print(f"  {mode:12s} Acc={s['mean_detection_accuracy']['mean']:.4f} "
              f"Energy={s['mean_energy']['mean']:.1f}J "
              f"Lifetime={s['mean_network_lifetime']['mean']:.2%} "
              f"(n={s[METRICS[0]]['n']})")
    with open(csv_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Table saved: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsl-kdd",
                        choices=["nsl-kdd", "unsw-nb15", "cicids2017"])
    parser.add_argument("--algorithm", type=str, default="dqn",
                        choices=["iql", "dqn", "maddpg", "qmix"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: all of config.SEEDS)")
    args = parser.parse_args()

    seeds = tuple(args.seeds) if args.seeds else tuple(config.SEEDS)
    run_ablation(args.dataset, seeds, algorithm=args.algorithm,
                 episodes=args.episodes)
