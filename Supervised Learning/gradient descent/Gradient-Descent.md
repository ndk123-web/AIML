# Linear Regression using Gradient Descent

This project implements **Linear Regression from scratch** using **Gradient Descent**, without using any machine learning libraries.
The goal is to understand **how m (slope) and b (intercept)** are learned by minimizing error.

---

## Problem Statement

Given a dataset of input–output pairs:

```
x → input feature
y → actual output (ground truth)
```

We want to find a straight line:

[
y = mx + b
]

such that the **prediction error is minimum**.

---

## Dataset Used

```python
x = [1, 2, 3, 4, 5]
y = [5, 7, 9, 11, 13]
```

This data roughly follows:

[
y = 2x + 3
]

But the algorithm **does not know this**.
It must discover the best `m` and `b` on its own.

---

## Model Assumption

We assume a linear model:

[
\hat{y} = mx + b
]

- `m` → slope of the line
- `b` → intercept
- `ŷ` → predicted output

---

## Loss Function (Cost Function)

To measure how bad our predictions are, we use **Mean Squared Error (MSE)**:

[
J(m, b) = \frac{1}{n} \sum (y_{\text{actual}} - y_{\text{predicted}})^2
]

Expanded form:

[
J(m, b) = \frac{1}{n} \sum (y - (mx + b))^2
]

### Why MSE?

- Penalizes large errors more
- Smooth curve → works well with gradient descent
- Differentiable

---

## Gradient Descent Idea

Gradient Descent updates `m` and `b` iteratively to **reduce the loss**.

Update rules:

[
m := m - \alpha \frac{\partial J}{\partial m}
]
[
b := b - \alpha \frac{\partial J}{\partial b}
]

Where:

- `α` = learning rate
- derivatives tell us **which direction reduces loss**

---

## Derivatives Used

From the loss function:

[
J(m, b) = \frac{1}{n} \sum (y - (mx + b))^2
]

We get:

### Derivative w.r.t. `m`

[
\frac{\partial J}{\partial m} = -\frac{2}{n} \sum x (y - \hat{y})
]

### Derivative w.r.t. `b`

[
\frac{\partial J}{\partial b} = -\frac{2}{n} \sum (y - \hat{y})
]

These are exactly what the code computes.

---

## Code Implementation

```python
import numpy as np

def gradientDescent(x, y):
    m = b = 0
    n = len(x)
    learningRate = 0.08
    iterations = 10000
    tolerance = 1e-6
    
    prevCost = float("inf")
    
    for i in range(iterations):
        yPredicted = m*x + b
        cost = (1/n) * sum((y - yPredicted) ** 2)
        
        md = -(2/n) * sum(x * (y - yPredicted))
        bd = -(2/n) * sum(y - yPredicted)
        
        m = m - learningRate * md
        b = b - learningRate * bd
        
        if abs(prevCost - cost) < tolerance:
            print("Converged at iteration", i)
            break
        
        prevCost = cost
        
        print(f"m {m}, b {b}, cost {cost}, iteration {i}")

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradientDescent(x, y)

```

---

## Dry Run (Iteration 0 – Fully Explained)

### Initial Values

```
m = 0
b = 0
```

### Step 1: Prediction

[
\hat{y} = 0·x + 0 = [0,0,0,0,0]
]

### Step 2: Error

[
y - \hat{y} = [5,7,9,11,13]
]

### Step 3: Cost

[
\text{cost} = \frac{1}{5}(25 + 49 + 81 + 121 + 169) = 89
]

### Step 4: Gradient for `m`

[
x·(y-\hat{y}) = [5,14,27,44,65]
]
[
\sum = 155
]
[
md = -\frac{2}{5}·155 = -62
]

### Step 5: Gradient for `b`

[
\sum(y-\hat{y}) = 45
]
[
bd = -\frac{2}{5}·45 = -18
]

### Step 6: Update Parameters

[
m = 0 - 0.08(-62) = 4.96
]
[
b = 0 - 0.08(-18) = 1.44
]

---

## What Happens Over Iterations?

- Each iteration:
  - Predictions improve
  - Cost decreases
  - Gradients shrink

- Eventually:
  - `md ≈ 0`
  - `bd ≈ 0`
  - Model converges

Final line approaches:

[
y ≈ 2x + 3
]

---

## Key Takeaways

- Gradient Descent **does not find a perfect line**
- It finds the **best-fit line with minimum error**
- Derivatives are taken of the **loss function**, not the line
- `x` and `y` are arrays → all data points contribute
- `m` and `b` are optimized iteratively

---

## One-Line Summary

> Linear Regression with Gradient Descent finds the values of `m` and `b` that minimize mean squared error by repeatedly moving parameters in the direction of lower loss.
