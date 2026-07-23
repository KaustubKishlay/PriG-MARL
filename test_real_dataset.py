"""
Manual smoke script -- prints values for a human to eyeball, no assertions.
For automated, assertion-based regression tests, see tests/ (run via
`python -m unittest discover -s tests`).
"""
import sys
sys.path.insert(0, ".")

from data.real_dataset import NSLKDDProcessor, RealTrafficEnvironment
from network.iot_network import IoTNetwork
from agents.baselines import RandomPolicy, AlwaysMonitorPolicy, evaluate_baseline
import numpy as np

# Load dataset
print("Loading NSL-KDD...")
dataset = NSLKDDProcessor(top_k=10, seed=42)
dataset.load()
stats = dataset.get_stats()
print(f"\nDataset loaded:")
print(f"  Train: {stats['train_samples']} samples ({stats['attack_ratio_train']:.1%} attacks)")
print(f"  Test:  {stats['test_samples']} samples ({stats['attack_ratio_test']:.1%} attacks)")
print(f"  Features: {stats['feature_names']}")

# Create environment
network = IoTNetwork(num_nodes=10, seed=42)
env = RealTrafficEnvironment(
    network=network, dataset=dataset,
    max_steps=50, use_test=True, seed=42,
)

print(f"\nEnvironment:")
print(f"  State dim: {env.state_dim} (10 traffic + 4 meta)")
print(f"  Nodes: {env.num_agents}")

# Test with baselines
print("\nRunning baselines on REAL data...")
for name, policy in [
    ("Random", RandomPolicy(10, seed=42)),
    ("AlwaysMonitor", AlwaysMonitorPolicy(10)),
]:
    network.reset()
    test_env = RealTrafficEnvironment(
        network=network, dataset=dataset,
        max_steps=50, use_test=True, seed=42,
    )
    result = evaluate_baseline(policy, test_env, 10, name)

print("\nReal dataset integration verified!")
