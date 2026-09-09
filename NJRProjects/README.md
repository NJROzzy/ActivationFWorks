# Sigmoid Activation

Exploration of the Sigmoid activation function — the first in our activation functions series.

## Purpose

Understand Sigmoid's mathematical behavior, its role historically in neural networks, and why it's largely been replaced in hidden layers of deep networks — while still remaining relevant in specific contexts (binary classification outputs, LSTM/GRU gates).

## Covers

- Mathematical definition and S-curve shape (range: 0 to 1)
- Derivative behavior and the vanishing gradient problem
- Zero-centering limitation and its effect on optimization
- From-scratch implementation (NumPy)
- `torch.nn` implementation (`nn.Sigmoid()` / `torch.sigmoid()`)
- Visualization of σ(x) and σ'(x) side by side

## Notes

Treat this as a living reference — implementation and visualizations will be added as exploration progresses.

