# PyTorch Deep Learning & NLP Fundamentals

A comprehensive guide and reference covering the foundational concepts, mathematical intuition, and implementation workflows in PyTorch based on the notebooks in this repository.

---

## Table of Contents
1. [Core PyTorch Philosophy & Architecture](#1-core-pytorch-philosophy--architecture)
2. [Tensors & Memory Model](#2-tensors--memory-model)
3. [Autograd & Dynamic Computation Graphs](#3-autograd--dynamic-computation-graphs)
4. [End-to-End Neural Network Pipeline](#4-end-to-end-neural-network-pipeline)
5. [Loss Functions & Objectives Comparison](#5-loss-functions--objectives-comparison)
6. [Model Saving, Loading & Serialization Formats](#6-model-saving-loading--serialization-formats)
   * [6.1 Standard `state_dict` (.pth / .pt)](#61-recommended-method-saving-state_dict-weights-only)
   * [6.2 Full Model Serialization (Pickle)](#62-full-model-serialization-pickle)
   * [6.3 Full Training Checkpoints](#63-full-training-checkpoints-resume-training)
   * [6.4 Modern Secure Serialization: SafeTensors (`.safetensors`)](#64-modern-secure-serialization-safetensors-safetensors)
7. [Comprehensive Concept Comparisons & Differences](#7-comprehensive-concept-comparisons--differences)
8. [Common Pitfalls & Best Practices](#8-common-pitfalls--best-practices)
9. [Notebook Index](#9-notebook-index)

---

## 1. Core PyTorch Philosophy & Architecture

PyTorch is designed around **Imperative Programming (Define-by-Run)** and **Object-Oriented Design (OOP)**:
* **Dynamic Computation Graphs**: The execution graph is constructed dynamically on-the-fly during the forward pass. Every iteration can have different branches, loops, or layer shapes.
* **Explicit Control Flow**: PyTorch avoids hidden abstractions. You explicitly write the forward pass, calculate loss, trigger backpropagation, step the optimizer, and clear accumulated gradients.
* **Hardware Agnostic Execution**: Code easily switches between CPU, CUDA (Nvidia GPUs), and MPS (Apple Silicon).

```python
import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
```

---

## 2. Tensors & Memory Model

Tensors are the fundamental data structures in PyTorch—multi-dimensional arrays optimized for hardware acceleration (GPU/TPU) and automatic differentiation.

### 2.1 Tensor Creation Methods
```python
import torch
import numpy as np

# 1. From Python lists
data = [[1, 2], [2, 3]]
x_data = torch.tensor(data, dtype=torch.float32)

# 2. From NumPy array (Zero-Copy memory sharing on CPU)
np_arr = np.array([[4, 5], [6, 7]])
x_np = torch.from_numpy(np_arr)

# 3. From existing tensor shapes (matching shape and dtype)
x_rand = torch.rand_like(x_data, dtype=torch.float32)

# 4. Standard shape-based initializers
shape = (2, 3)
rand_t = torch.rand(shape)    # Uniform distribution [0, 1)
ones_t = torch.ones(shape)    # Filled with 1.0
zeros_t = torch.zeros(shape)  # Filled with 0.0
```

### 2.2 Key Tensor Attributes
* `tensor.shape` or `tensor.size()`: Dimensionality of the tensor.
* `tensor.dtype`: Data type (e.g., `torch.float32`, `torch.int64`).
* `tensor.device`: Location in memory (`cpu`, `cuda:0`).
* `tensor.requires_grad`: Boolean tracking whether autograd records operations on this tensor.

---

## 3. Autograd & Dynamic Computation Graphs

PyTorch uses an automatic differentiation engine called **Autograd** which creates a directed acyclic graph (DAG) of operations during the forward pass.

### 3.1 Mathematical Intuition & The Chain Rule
Consider the mathematical system:
$$a = x^2$$
$$b = 2a$$
$$c = 3y$$
$$z = b + c + 5$$

Using the multivariable chain rule:
$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial b} \cdot \frac{\partial b}{\partial a} \cdot \frac{\partial a}{\partial x} = 1 \cdot 2 \cdot (2x) = 4x$$
For $x = 3.0$:
$$\frac{\partial z}{\partial x} = 4 \times 3.0 = 12.0$$

$$\frac{\partial z}{\partial y} = \frac{\partial z}{\partial c} \cdot \frac{\partial c}{\partial y} = 1 \cdot 3 = 3.0$$

### 3.2 Graph Execution in PyTorch
```python
x = torch.tensor(3.0, requires_grad=True)  # Leaf tensor
y = torch.tensor(4.0, requires_grad=True)  # Leaf tensor

# Forward pass
a = x ** 2          # a = 9.0  (grad_fn: <PowBackward0>)
a.retain_grad()     # Retain intermediate non-leaf gradient for inspection

b = 2 * a           # b = 18.0 (grad_fn: <MulBackward0>)
b.retain_grad()

c = 3 * y           # c = 12.0 (grad_fn: <MulBackward0>)
c.retain_grad()

z = b + c + 5       # z = 35.0 (grad_fn: <AddBackward0>)
z.retain_grad()

# Backward pass (computes gradients via reverse-mode autodiff)
z.backward(retain_graph=True)

print("dz/dx:", x.grad)  # 12.0
print("dz/dy:", y.grad)  # 3.0
print("dz/da:", a.grad)  # 2.0
print("dz/db:", b.grad)  # 1.0
print("dz/dc:", c.grad)  # 1.0
```

### 3.3 Leaf Tensors vs Intermediate Tensors
* **Leaf Tensors**: Created directly by the user (or model parameters) with `requires_grad=True`. Their `.grad_fn` is `None`, and their gradients are persisted in `.grad` after `.backward()`.
* **Intermediate (Non-Leaf) Tensors**: Generated as outputs of operations. Their `.grad_fn` points to the backward function that generated them. PyTorch automatically deallocates intermediate gradients after `.backward()` to conserve memory unless `.retain_grad()` is explicitly called.

---

## 4. End-to-End Neural Network Pipeline

Building models in PyTorch follows an Object-Oriented paradigm by subclassing `nn.Module`.

### 4.1 Model Definition
```python
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten input: (Batch, 1, 28, 28) -> (Batch, 784)
        self.flatten = nn.Flatten()
        
        # Sequential container chaining layers and non-linearities
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # Z1 = X * W1 + b1
            nn.ReLU(),
            nn.Linear(512, 512),      # Z2 = A1 * W2 + b2
            nn.ReLU(),
            nn.Linear(512, 10)        # Logits: raw unnormalized scores
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
```

### 4.2 The 5-Step Training Loop
Every training step executes the exact sequence:

```
[1. Forward Pass] -> [2. Compute Loss] -> [3. Backward Pass] -> [4. Update Weights] -> [5. Zero Gradients]
   pred = model(X)      loss = loss_fn(pred, y)   loss.backward()         optimizer.step()        optimizer.zero_grad()
```

```python
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()  # Sets training mode (activates Dropout / BatchNorm tracking)
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # 2. Backward Pass
        loss.backward()        # Computes dLoss/dW and dLoss/db for all parameters

        # 3. Optimizer Step
        optimizer.step()       # Updates weights: W = W - lr * W.grad

        # 4. Zero Gradients
        optimizer.zero_grad()  # Clears gradients before the next iteration
```

### 4.3 Evaluation / Testing Loop
```python
def test_loop(dataloader, model, loss_fn, device):
    model.eval()  # Sets evaluation mode (freezes Dropout / BatchNorm stats)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0.0

    # Disable autograd tracking to save memory and compute
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss:>8f}")
```

---

## 5. Loss Functions & Objectives Comparison

Choosing the correct loss function depends on the objective type and output expectations:

| Objective Type | Task Description | PyTorch Loss Function | Expected Model Output ($\hat{y}$) | Target Format ($y$) | Mathematical Formula |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regression** | Predicting continuous values (e.g., house price, temperature) | `nn.MSELoss()`<br>`nn.L1Loss()` | Continuous floats $\hat{y} \in \mathbb{R}$ | Continuous floats $y \in \mathbb{R}$ | $\text{MSE} = \frac{1}{N}\sum (\hat{y}_i - y_i)^2$<br>$\text{MAE} = \frac{1}{N}\sum \|\hat{y}_i - y_i\|$ |
| **Binary Classification** | Two classes (0 or 1) (e.g., spam vs ham, sentiment) | `nn.BCEWithLogitsLoss()` | Raw scalar logits $\hat{y} \in (-\infty, +\infty)$ | Float labels $y \in \{0.0, 1.0\}$ | Combines Sigmoid + BCE with numeric stability |
| **Multi-class Classification** | $C > 2$ mutually exclusive classes (e.g., MNIST, CIFAR) | `nn.CrossEntropyLoss()` | Raw unnormalized logits $\hat{y} \in \mathbb{R}^{B \times C}$ | Integer Class Indices $y \in \{0, \dots, C-1\}$ | Combines `LogSoftmax` + `NLLLoss` |
| **Multi-label Classification** | Multiple non-exclusive binary attributes per sample | `nn.BCEWithLogitsLoss()` | Logits per class $\hat{y} \in \mathbb{R}^{B \times C}$ | Float binary matrix $y \in \{0.0, 1.0\}^{B \times C}$ | Applied independently per label |

> **Crucial Rule on `nn.CrossEntropyLoss()`**: Do **not** pass a `nn.Softmax()` output into `nn.CrossEntropyLoss()`. It expects raw unnormalized logits. Applying Softmax beforehand will double-apply softmax and degrade gradient dynamics.

---

## 6. Model Saving, Loading & Serialization Formats

Real-world machine learning architectures separate **Parameters/Weights** from **Architecture/Configuration**:

```
Model Artifacts:
├── Weights / Tensors: model.pth / model.safetensors (Trained W and b)
└── Architecture / Config: Python Class / config.json (Layer counts, dimensions)
```

### 6.1 Recommended Method: Saving `state_dict` (Weights Only)
Saves only learned parameters as an ordered dictionary. Decoupled from Python class pickling, portable, and secure.

```python
# 1. Save Weights
torch.save(model.state_dict(), "model_weights.pth")

# 2. Load Weights
loaded_model = NeuralNetwork()                                  # Instantiate architecture
loaded_model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
loaded_model.eval()                                            # Set to inference mode
```

### 6.2 Full Model Serialization (Pickle)
Saves the entire Python object including class code bindings.
```python
# Save full model
torch.save(model, "full_model.pth")

# Load full model (requires the exact class definition in scope)
loaded_model = torch.load("full_model.pth")
```
* **Drawback**: Fragile across directory restructuring, class refactoring, or differing Python/PyTorch versions.

### 6.3 Full Training Checkpoints (Resume Training)
When training large models across long intervals, checkpoints store optimizer states, epochs, and loss along with weights:

```python
# 1. Save Checkpoint
checkpoint = {
    "epoch": current_epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": current_loss,
}
torch.save(checkpoint, "checkpoint.pth")

# 2. Resume from Checkpoint
checkpoint = torch.load("checkpoint.pth", weights_only=False)
model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]
```

### 6.4 Modern Secure Serialization: SafeTensors (`.safetensors`)
**SafeTensors** is a modern format developed by Hugging Face to store tensors safely (no arbitrary code execution) and quickly (zero-copy memory mapping):

```python
from safetensors.torch import save_file, load_file

# 1. Save weights with safetensors
save_file(model.state_dict(), "model.safetensors")

# 2. Load weights with safetensors
state_dict = load_file("model.safetensors")
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(state_dict)
loaded_model.eval()
```

#### Why SafeTensors?
1. **Security**: PyTorch's native `.pt` format relies on Python `pickle`, which can execute arbitrary malicious code upon loading. `safetensors` contains **zero executable code**—only raw tensor byte buffers and a JSON metadata header.
2. **Speed & Memory-Mapping (`mmap`)**: Allows zero-copy loading directly from disk into GPU/CPU RAM, enabling instant loads and parameter slicing without loading the entire weight file into system memory.
3. **Cross-Framework Support**: Interoperable across PyTorch, TensorFlow, JAX, Rust, and C++.

---

## 7. Comprehensive Concept Comparisons & Differences

### Comparison 1: `.pth` / `.pt` (PyTorch Pickle) vs `.safetensors` (SafeTensors)

| Aspect | PyTorch `.pt` / `.pth` | Hugging Face `.safetensors` |
| :--- | :--- | :--- |
| **Serialization Engine** | Python `pickle` (or PyTorch ZIP container) | Pure flat binary tensor buffers + JSON metadata |
| **Security Risk** | **High** (Arbitrary code execution exploit possible via pickle) | **Zero** (Completely safe from arbitrary code execution) |
| **Loading Speed** | Slower (deserializes Python object graphs) | **Extremely fast** (direct memory mapping / zero-copy) |
| **Partial Loading** | Must deserialize the full dictionary into RAM | Allows slicing & loading specific weight subsets |
| **Cross-Language Support**| Primarily Python-centric | Language agnostic (Rust, C++, Python, WebAssembly) |
| **Hugging Face / LLM Standard** | Legacy format | **Primary standard** for modern LLMs & Transformers |

---

### Comparison 2: `state_dict` vs Full Model vs Checkpoint vs SafeTensors

| Feature | `state_dict` (`.pth`) | Full Model Pickle (`.pth`) | Full Checkpoint (`.pth`) | SafeTensors (`.safetensors`) |
| :--- | :--- | :--- | :--- | :--- |
| **What is saved?** | Parameter weights & biases | Python object + Architecture + Weights | Weights + Optimizer State + Epoch + Loss | Parameter weights & biases (flat buffers) |
| **Portability** | High | Low (breaks on code refactor) | High (training resumption) | **Highest** (cross-framework) |
| **Arbitrary Code Execution** | Possible if unpickling untrusted files | Highly vulnerable | Possible | **Impossible** |
| **Primary Use Case** | Native PyTorch deployment | Quick local debugging | Resuming interrupted training | Production NLP/LLM deployment |

---

### Comparison 3: `model.train()` vs `model.eval()`

| Feature | `model.train()` | `model.eval()` |
| :--- | :--- | :--- |
| **Dropout** | Randomly zeros out neurons to prevent co-adaptation | Deactivated (all neurons active, scaled appropriately) |
| **Batch Normalization** | Updates running mean & variance using batch stats | Freezes running statistics; uses saved population stats |
| **Gradients** | Still computed (unless inside `torch.no_grad()`) | Still computed if backward is called |
| **When to use** | During forward passes of training epochs | During validation, testing, and production inference |

---

### Comparison 4: `with torch.no_grad():` vs `tensor.detach()`

| Aspect | `with torch.no_grad():` | `tensor.detach()` |
| :--- | :--- | :--- |
| **Scope** | Context manager affecting all operations inside the block | Method operating on an individual tensor |
| **Mechanism** | Disables computation graph tracking entirely for all ops | Creates a new tensor detached from the current graph |
| **Memory Impact** | Saves memory for intermediate tensors across the block | Keeps original tensor in memory while creating a detached view |
| **Primary Use Case** | Validation / Test loops and model evaluation | Extracting predictions, plotting numpy conversions, RL replay buffers |

---

### Comparison 5: `torch.Tensor` vs `np.ndarray`

| Feature | PyTorch Tensor (`torch.Tensor`) | NumPy Array (`np.ndarray`) |
| :--- | :--- | :--- |
| **Hardware Acceleration** | Native GPU (CUDA), MPS, and TPU acceleration | CPU execution only |
| **Automatic Differentiation** | Built-in via Autograd engine (`requires_grad`) | None (requires manual gradient implementation) |
| **Interoperability** | `torch.from_numpy(arr)` / `t.numpy()` | Zero-copy memory sharing on CPU |
| **Deep Learning Ecosystem** | Integrates with `torch.nn`, `DataLoader`, TorchScript | Standard for scientific computing and feature engineering |

---

### Comparison 6: `optimizer.zero_grad()` vs `optimizer.zero_grad(set_to_none=True)`

| Option | Behavior | Memory & Performance |
| :--- | :--- | :--- |
| `optimizer.zero_grad()` | Overwrites `param.grad` tensors with zeros (`0.0`) | Keeps gradient buffers allocated; standard behavior |
| `optimizer.zero_grad(set_to_none=True)` | Sets `param.grad = None` | Deallocates memory buffers, faster execution, lower memory footprint |

---

## 8. Common Pitfalls & Best Practices

1. **Forgetting `optimizer.zero_grad()`**:
   * *Problem*: In PyTorch, gradients accumulate by default (`param.grad += dL/dW`). If you do not call `zero_grad()`, gradients from the current batch add to gradients from previous batches, causing explosive updates.
   * *Fix*: Always call `optimizer.zero_grad()` before or immediately after `optimizer.step()`.

2. **Mixing Softmax with `nn.CrossEntropyLoss`**:
   * *Problem*: `nn.CrossEntropyLoss` already includes `LogSoftmax`. Putting a `nn.Softmax()` in the model leads to numerical instability and poor convergence.
   * *Fix*: Output raw linear logits from your final layer.

3. **Memory Leaks from Tracking Loss**:
   * *Problem*: Writing `total_loss += loss` retains the entire computation graph in memory for the whole epoch.
   * *Fix*: Use `total_loss += loss.item()` to extract the scalar float value.

4. **Missing `model.eval()` and `torch.no_grad()` during inference**:
   * *Problem*: Dropout stays active and Autograd consumes massive memory storing intermediate states.
   * *Fix*: Always wrap evaluation code with `model.eval()` and `with torch.no_grad():`.

5. **Loading Untrusted `.pth` Files vs `.safetensors`**:
   * *Problem*: Loading arbitrary `.pt` / `.pth` files from the internet can trigger arbitrary code execution via Python pickle exploits.
   * *Fix*: Use `weights_only=True` with `torch.load()` or adopt `.safetensors` via `safetensors.torch.load_file()`.

---

## 9. Notebook Index
* [1_QuickStart.ipynb](file:///d:/ML/ML-Learning/NLP/Pytorch/1_QuickStart.ipynb): Full FashionMNIST classification pipeline from scratch.
* [2_Tensors.ipynb](file:///d:/ML/ML-Learning/NLP/Pytorch/2_Tensors.ipynb): Tensor initialization, conversions, and tensor math.
* [3_comp_graph_n_required_grad.ipynb](file:///d:/ML/ML-Learning/NLP/Pytorch/3_comp_graph_n_required_grad.ipynb): Dynamic computation graph, `requires_grad`, Autograd, and `retain_grad()`.
* [4_save_n_load_models.ipynb](file:///d:/ML/ML-Learning/NLP/Pytorch/4_save_n_load_models.ipynb): Regression model, `state_dict`, full checkpoints, and inference.
* [load_n_save_safetensor.ipynb](file:///d:/ML/ML-Learning/NLP/Pytorch/load_n_save_safetensor.ipynb): Safe tensor serialization with Hugging Face `safetensors` (`save_file`, `load_file`).