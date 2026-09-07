# Activation Functions

Collaborative exploration of activation functions in neural networks, covering the range of functions available in `torch.nn` (and beyond) — from foundational (ReLU, Sigmoid, Tanh) to more specialized/modern variants (GELU, SiLU, Mish, etc.).

## Purpose

This folder is a joint deep-dive into *why* activation functions matter, not just *what* they do. Goals include:

- Understanding the mathematical behavior of each function (shape, range, derivative)
- Comparing gradient flow characteristics (e.g. vanishing/exploding gradients)
- Identifying practical use cases and trade-offs (e.g. why GELU is preferred in transformers, why ReLU dominates CNNs)
- Hands-on implementation — both from scratch (plain Python/NumPy) and using `nn` built-ins
- Visualizing function curves and their derivatives side by side

## Scope

Exploration spans the ~400 activation functions available in `torch.nn` and related libraries, prioritized by relevance and common usage rather than exhaustive coverage of every variant.

## Approach

Work here follows the same style as the rest of the collaboration: concepts explained first, hands-on implementation second. Findings, comparisons, and notebooks are split between individual exploration and joint synthesis as understanding develops.

## Notes

This folder is expected to evolve as functions are explored and comparisons are added — treat it as a living reference rather than a fixed structure.