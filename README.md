# ActivationFWorks

This repository is a compact reference for common **neural-network activation functions** and how they are typically used across different model families.

## Activation functions

### 1) Sigmoid
- Formula: `σ(x) = 1 / (1 + e^-x)`
- Output range: `(0, 1)`
- Common use: binary output probabilities.

### 2) Tanh
- Formula: `tanh(x) = (e^x - e^-x) / (e^x + e^-x)`
- Output range: `(-1, 1)`
- Common use: hidden layers when centered outputs are useful.

### 3) ReLU
- Formula: `ReLU(x) = max(0, x)`
- Output range: `[0, ∞)`
- Common use: default hidden-layer activation in many deep networks.

### 4) Leaky ReLU
- Formula: `LeakyReLU(x) = max(ax, x)` where `a` is small (e.g., `0.01`)
- Output range: `(-∞, ∞)`
- Common use: mitigates “dying ReLU” behavior.

### 5) ELU
- Formula: `ELU(x) = x` if `x > 0`, otherwise `a(e^x - 1)`
- Output range: `(-a, ∞)`
- Common use: smoother negative branch than ReLU family.

### 6) Softmax
- Formula: `softmax(x_i) = e^{x_i} / Σ_j e^{x_j}`
- Output range: `(0, 1)` and sums to `1` across classes.
- Common use: multi-class output layers.

## Typical model + activation pairings

- **MLP / Feed-forward networks**: ReLU/Leaky ReLU in hidden layers, Sigmoid or Softmax in output.
- **CNNs**: ReLU-family activations in convolution blocks, Softmax/Sigmoid at output depending on task.
- **RNN/LSTM/GRU**: Tanh and Sigmoid gates are standard; task-dependent output activations.
- **Transformers**: ReLU-family alternatives (often GELU in practice) in feed-forward blocks, Softmax for attention/output distributions.

## Quick selection guide

- Use **ReLU** as a strong baseline for hidden layers.
- Use **Sigmoid** for binary classification output.
- Use **Softmax** for single-label multi-class output.
- Consider **Leaky ReLU/ELU** if ReLU units stop activating.