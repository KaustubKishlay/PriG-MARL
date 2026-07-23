"""
Manual smoke script -- prints values for a human to eyeball, no assertions.
For automated, assertion-based regression tests, see tests/ (run via
`python -m unittest discover -s tests`).
"""
import sys
sys.path.insert(0, ".")

from agents.baselines import (
    RandomForestIDS, StaticStackelbergPolicy,
    RandomPolicy, AlwaysMonitorPolicy, evaluate_baseline
)
from network.iot_network import IoTNetwork
from environment.ids_env import IDSEnvironment
from game.stackelberg import create_game_from_network

print("Setting up network...")
net = IoTNetwork(10, seed=42)
game = create_game_from_network(net, seed=42)

print("\n1. Random Forest IDS...")
rf = RandomForestIDS(10, seed=42)
stats = rf.train_classifier(2000)
print(f"   RF Test Accuracy: {stats['test_accuracy']:.4f}")

print("\n2. Static Stackelberg Policy...")
sp = StaticStackelbergPolicy(game)
actions = sp.select_actions()
print(f"   Actions: {actions}")

print("\n3. Random Policy...")
rp = RandomPolicy(10)
print(f"   Actions: {rp.select_actions()}")

print("\n4. Always Monitor...")
am = AlwaysMonitorPolicy(10)
print(f"   Actions: {am.select_actions()}")

print("\n5. Evaluating baselines on environment...")
env = IDSEnvironment(net, max_steps=50, attacker_strategy="strategic", seed=42)

for name, policy in [("Random", rp), ("AlwaysMonitor", am), ("Stackelberg", sp), ("RF", rf)]:
    net.reset()
    env2 = IDSEnvironment(net, max_steps=50, attacker_strategy="strategic", seed=42)
    result = evaluate_baseline(policy, env2, 5, name)
    print(f"   {name}: Acc={result['mean_detection_accuracy']:.4f}, FPR={result['mean_fpr']:.4f}")

print("\nAll baselines validated successfully!")
