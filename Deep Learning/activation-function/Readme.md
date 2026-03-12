# Activation Functions

## What is an Activation Function?

An activation function decides whether a neuron should be activated ("fire") or not. It introduces non-linearity into the neural network, allowing it to learn complex patterns.

**Why do we need them?**
- Without activation functions, neural networks would just be linear models
- They help networks learn non-linear relationships in data
- They determine what information gets passed to the next layer

## Popular Activation Functions

### 1. **Sigmoid Function**

**Formula:** σ(x) = 1 / (1 + e^(-x))

**Output Range:** 0 to 1

**When to Use:**
- **Binary Classification** (output layer)
- When you need probability values
- When outputs must be between 0 and 1

**Advantages:**
- Smooth gradient
- Output is probability-like
- Good for binary decisions

**Disadvantages:**
- **Vanishing Gradient Problem**: Gradients get very small for extreme values
- Slow convergence
- Not zero-centered (outputs always positive)
- Computationally expensive (contains exponential)

---

### 2. **Tanh (Hyperbolic Tangent)**

**Formula:** tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**Output Range:** -1 to 1

**When to Use:**
- Hidden layers (better than sigmoid for hidden layers)
- When you need zero-centered outputs

**Advantages:**
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid

**Disadvantages:**
- Still suffers from **vanishing gradient problem**
- Computationally expensive

---

### 3. **ReLU (Rectified Linear Unit)** ⭐ Most Popular

**Formula:** ReLU(x) = max(0, x) = { x if x > 0, 0 if x ≤ 0 }

**Output Range:** 0 to ∞

**When to Use:**
- **Hidden layers** (default choice for most networks)
- Convolutional Neural Networks (CNNs)
- Deep networks

**Advantages:**
- **Very fast computation** (simple max operation)
- **Lightweight** (no exponentials)
- Helps with vanishing gradient (for positive values)
- Sparse activation (some neurons output zero)
- Converges faster than sigmoid/tanh

**Disadvantages:**
- **Dying ReLU problem**: Neurons can get stuck outputting 0
- Not zero-centered
- Unbounded output (can explode)

---

### 4. **Leaky ReLU**

**Formula:** Leaky ReLU(x) = { x if x > 0, αx if x ≤ 0 }

(where α is a small constant like 0.01)

**Output Range:** -∞ to ∞

**When to Use:**
- When you face dying ReLU problem
- Hidden layers as an alternative to ReLU

**Advantages:**
- Solves the **dying ReLU problem**
- Allows small negative values (keeps gradient flow)
- Fast computation

**Disadvantages:**
- Need to choose α hyperparameter
- Sometimes inconsistent performance

---

## Quick Selection Guide

| Layer Type | Recommended Activation | Why? |
|-----------|----------------------|------|
| **Output Layer** (Binary) | Sigmoid | Gives probability (0-1) |
| **Output Layer** (Multi-class) | Softmax | Gives probability distribution |
| **Output Layer** (Regression) | Linear/None | Unbounded output |
| **Hidden Layers** | **ReLU** | Fast, efficient, works well |
| **Hidden Layers** (if ReLU fails) | Leaky ReLU or ELU | Solves dying ReLU |

## Common Problems

### 1. **Vanishing Gradient Problem**
- **Affects:** Sigmoid, tanh, and sometimes ReLU
- **What happens:** Gradients become too small during backpropagation
- **Result:** Deep layers don't learn effectively
- **Solution:** Use ReLU or its variants

### 2. **Dying ReLU Problem**
- **Affects:** ReLU
- **What happens:** Neurons output 0 for all inputs and stop learning
- **Result:** Part of the network becomes inactive
- **Solution:** Use Leaky ReLU, ELU, or lower learning rate

## Best Practices

1. **Start with ReLU** for hidden layers (it works 90% of the time)
2. **Use Sigmoid** only for binary classification output
3. **Use Softmax** for multi-class classification output
4. **Try Leaky ReLU** if you notice dead neurons
5. **Avoid sigmoid/tanh** in hidden layers of deep networks