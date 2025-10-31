# 🧠 PhysNet — Differentiable Physics Neural Network

(Vibecoded. Ask from AI to explain it to you. )

Version: 0.1 — Research Prototype
License: MIT

# Overview

PhysNet is a proof-of-concept neural network that replaces the usual dense hidden layers with a differentiable physics simulation — a small 2D field evolved by a Gross–Pitaevskii-style equation.
It learns standard image tasks while keeping a modest ability to retain previously learned tasks, showing potential for catastrophic forgetting reduction through physical attractor dynamics.

# Key Features

Differentiable GPE physics core acting as an internal “field memory.”

End-to-end training on image classification tasks (MNIST, permuted MNIST).

Achieves strong single-task performance (≈98–99% on MNIST).

Retains ≈70–75% accuracy on the first task after learning a conflicting second task (~25–30% forgetting).

Implemented in pure PyTorch — no external solvers needed.

# Quick Start

git clone https://github.com/anttiluode/physnet.git

cd physnet

pip install -r requirements.txt

python neuron_grok5.py

# Runs two tasks in sequence:

Task 1: Standard MNIST

Task 2: Permuted MNIST

Then tests how much of Task 1 performance remains.

# Example Output

Task 1 Accuracy: 98.86%

Task 2 Accuracy: 95.93%

Task 1 After Task 2: 72.43%

Catastrophic Forgetting: 26.43%

# Why It Matters

While state-of-the-art continual learning models can reach as low as 5% forgetting, PhysNet introduces a different mechanism — not regularization or replay, but differentiable physical memory.
This prototype suggests that attractor dynamics may offer a path toward more biologically and physically grounded continual learning systems.

# Next Steps

Add more tasks (3+ permuted datasets)

Compare against EWC / SI baselines

Visualize field attractors over training
Try the physics core as a plug-in module in other architectures

# Example results

Using device: cuda
Loading data...

--- TRAINING ON TASK 1 (Normal MNIST) ---
Epoch 1/5
Epoch 2/5
Epoch 3/5
Epoch 4/5
Epoch 5/5
Test set: Avg. loss: 0.0431, Accuracy: 9886/10000 (98.86%)

Final Task 1 Accuracy (after Task 1): 98.86%

--- TRAINING ON TASK 2 (Permuted MNIST) ---
Epoch 1/5
Epoch 2/5
Epoch 3/5
Epoch 4/5
Epoch 5/5
Test set: Avg. loss: 0.1354, Accuracy: 9593/10000 (95.93%)

Final Task 2 Accuracy (after Task 2): 95.93%

--- FORGETTING TEST (Testing Task 1 again) ---
Test set: Avg. loss: 1.1666, Accuracy: 7243/10000 (72.43%)

--- 🔥 RESULTS 🔥 ---
Task 1 (Original):   98.86%
Task 2 (Permuted):   95.93%
Task 1 (After T2): 72.43%
Catastrophic Forgetting: 26.43%

🧠 Note: Model accuracy is high, but forgetting (26.43%) is also significant.

