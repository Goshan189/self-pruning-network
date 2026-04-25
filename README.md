# Self-Pruning Neural Network

## Overview

This project implements a self-pruning neural network where each weight is associated with a learnable gate. During training, the network learns to suppress unimportant connections, resulting in a sparse and efficient architecture.

The model is trained on the CIFAR-10 dataset using a custom pruning mechanism integrated directly into the training process.

---

## Method

### Prunable Layer

Each weight in the network is paired with a learnable gate parameter:

pruned_weight = weight × sigmoid(gate_score)

The sigmoid function ensures that gate values remain between 0 and 1. When a gate approaches 0, the corresponding weight is effectively removed from the network.

---

### Loss Function

The training objective combines classification performance with sparsity:

Total Loss = CrossEntropyLoss + λ × SparsityLoss

Where:

SparsityLoss = Σ gates

---

## Why L1 Regularization Encourages Sparsity

The sparsity loss is defined as the L1 norm (sum) of the gate values. Since gate values lie between 0 and 1, minimizing this term pushes many gates toward zero.

Unlike L2 regularization, which reduces values gradually, L1 regularization promotes exact or near-zero values. As a result, many connections become inactive, leading to a sparse network.

---

## Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
| ------ | ----------------- | ------------ |
| 0.01   | 53.94             | 13.42        |
| 0.1    | 53.68             | 27.65        |
| 0.5    | 54.36             | 43.08        |

---

## Analysis

We observe that increasing λ leads to a consistent increase in sparsity, demonstrating that the network successfully learns to prune its connections.

Interestingly, test accuracy remains stable across all λ values and slightly improves at higher sparsity levels. This suggests that the original network contains redundant parameters, and pruning removes unnecessary connections without degrading performance.

This indicates that the model is over-parameterized, and the pruning mechanism acts as an implicit regularizer, improving efficiency while maintaining predictive capability.

Beyond a certain point, further increases in λ are expected to reduce accuracy as important connections begin to be removed.

---

## Gate Distribution

The distribution of gate values shows a strong concentration near zero, indicating that many connections have been effectively pruned. At the same time, a subset of gates remains active, preserving important features required for accurate predictions.

This confirms that the model successfully learns a sparse representation while retaining essential information.
