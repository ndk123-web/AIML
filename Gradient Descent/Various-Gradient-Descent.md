# Gradient Descent Variants in Machine Learning

Gradient Descent is an optimization algorithm used to minimize the loss function by updating model weights iteratively.

Weight update rule:

[
w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}
]

Where:

- **w** = model weights
- **η (learning rate)** = step size
- **L** = loss function
- **∂L/∂w** = gradient (slope of the loss)

The difference between Gradient Descent variants depends on **how much data is used to compute the gradient before updating weights.**

---

# 1. Batch Gradient Descent

### Idea

Uses the **entire dataset** to compute the gradient before updating weights.

### Workflow

```
Dataset → Forward Pass → Compute Loss → Compute Gradient → Update Weights
```

### Characteristics

| Property          | Value                   |
| ----------------- | ----------------------- |
| Samples used      | Entire dataset          |
| Updates per epoch | 1                       |
| Stability         | High                    |
| Speed             | Slow for large datasets |

### Example

Dataset size = 1000

```
Step 1 → use 1000 samples → compute gradient
Step 2 → update weights
```

Only **1 update per epoch**.

---

### Code Example

```python
import numpy as np

X = np.array([1,2,3,4])
y = np.array([2,4,6,8])

w = 0
lr = 0.01

for epoch in range(100):

    y_pred = w * X
    loss = np.mean((y - y_pred) ** 2)

    gradient = -2 * np.mean(X * (y - y_pred))

    w = w - lr * gradient

print("Weight:", w)
```

---

# 2. Stochastic Gradient Descent (SGD)

### Idea

Uses **one sample at a time** to update weights.

### Workflow

```
Sample → Forward Pass → Loss → Gradient → Update Weights
```

### Characteristics

| Property          | Value                 |
| ----------------- | --------------------- |
| Samples used      | 1                     |
| Updates per epoch | n (number of samples) |
| Speed             | Fast                  |
| Stability         | Noisy updates         |

### Example

Dataset = 1000 samples

```
sample1 → update
sample2 → update
sample3 → update
...
sample1000 → update
```

Total **1000 updates per epoch**.

---

### Code Example

```python
import numpy as np

X = np.array([1,2,3,4])
y = np.array([2,4,6,8])

w = 0
lr = 0.01

for epoch in range(100):

    for i in range(len(X)):

        x_i = X[i]
        y_i = y[i]

        y_pred = w * x_i

        gradient = -2 * x_i * (y_i - y_pred)

        w = w - lr * gradient

print("Weight:", w)
```

---

# 3. Mini-Batch Gradient Descent

### Idea

Uses a **small subset (batch)** of data to compute gradients.

### Workflow

```
Batch → Forward Pass → Loss → Gradient → Update Weights
```

### Characteristics

| Property          | Value                         |
| ----------------- | ----------------------------- |
| Samples used      | Batch size (e.g. 32, 64, 128) |
| Updates per epoch | dataset_size / batch_size     |
| Stability         | Balanced                      |
| Speed             | Fast                          |

This is the **most commonly used method in Deep Learning**.

---

### Example

Dataset = 1000 samples
Batch size = 100

```
Batch1 (100 samples) → update
Batch2 (100 samples) → update
Batch3 (100 samples) → update
...
Batch10 → update
```

Total **10 updates per epoch**.

---

### Code Example

```python
import numpy as np

X = np.array([1,2,3,4,5,6,7,8])
y = 2 * X

w = 0
lr = 0.01
batch_size = 2

for epoch in range(100):

    for i in range(0, len(X), batch_size):

        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        y_pred = w * X_batch

        gradient = -2 * np.mean(X_batch * (y_batch - y_pred))

        w = w - lr * gradient

print("Weight:", w)
```

---

# Visual Comparison

| Method                      | Samples used   | Updates per epoch |
| --------------------------- | -------------- | ----------------- |
| Batch Gradient Descent      | Entire dataset | 1                 |
| Stochastic Gradient Descent | 1 sample       | n                 |
| Mini Batch Gradient Descent | Small batch    | n / batch_size    |

---

# Practical Usage

Modern Deep Learning frameworks use:

```
Mini-Batch Gradient Descent
```

Common batch sizes:

```
32
64
128
256
```

Frameworks using this approach:

- TensorFlow
- PyTorch
- Keras

---

# Key Insight

All three methods use the same weight update formula:

[
w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}
]

The only difference is **how much data is used to compute the gradient before updating weights.**

---
