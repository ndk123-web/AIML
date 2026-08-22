# Neural Network Backpropagation (Step-by-Step with Example)

## 1. Why this matters

In a neural network, the final output depends on many layers of transformations. During training, we need to adjust all weights so that the predicted output becomes closer to the actual target. The challenge is that early-layer weights affect the output indirectly through multiple intermediate layers.

Backpropagation solves this by computing how much each weight contributed to the final error using the chain rule.

---

## 2. Problem Setup

We define a simple neural network:

* Input layer: 4 features → a, b, c, d
* Hidden layer: 2 neurons → x, y
* Output layer: 2 neurons → o1, o2
* Loss function: Mean Squared Error (MSE)

### Architecture

[a b c d] → [x y] → [o1 o2]

---

## 3. Forward Pass

### Step 1: Hidden Layer Computation

Each hidden neuron takes all input features:

- x = w1*a + w2*b + w3*c + w4*d + b1
- y = w5*a + w6*b + w7*c + w8*d + b2

This is not splitting features. Each neuron learns a different combination of all inputs.

---

### Step 2: Output Layer Computation

- o1 = v1*x + v2*y + b3
- o2 = v3*x + v4*y + b4

---

### Step 3: Loss Function

We use Mean Squared Error:

L = (1/2) * [(o1 - y1)^2 + (o2 - y2)^2]

where:

* y1, y2 are actual target values

---

## 4. Numerical Example

Assume:

Inputs:
a = 1, b = 2, c = 3, d = 4

Weights (input → hidden):
w1=0.1, w2=0.2, w3=0.3, w4=0.4
w5=0.5, w6=0.6, w7=0.7, w8=0.8

Bias:
b1=0.5, b2=0.5

Weights (hidden → output):
v1=0.9, v2=1.0
v3=1.1, v4=1.2

Bias:
b3=0.5, b4=0.5

Target:
y1 = 10, y2 = 12

---

### Step 1: Compute Hidden Layer

x = (0.1*1 + 0.2*2 + 0.3*3 + 0.4*4) + 0.5
= (0.1 + 0.4 + 0.9 + 1.6) + 0.5
= 3.5

y = (0.5*1 + 0.6*2 + 0.7*3 + 0.8*4) + 0.5
= (0.5 + 1.2 + 2.1 + 3.2) + 0.5
= 7.5

---

### Step 2: Compute Output Layer

o1 = (0.9*3.5 + 1.0*7.5) + 0.5
= (3.15 + 7.5) + 0.5
= 11.15

o2 = (1.1*3.5 + 1.2*7.5) + 0.5
= (3.85 + 9.0) + 0.5
= 13.35

---

### Step 3: Compute Loss

L = (1/2)*[(11.15 - 10)^2 + (13.35 - 12)^2]
= (1/2)*[(1.15)^2 + (1.35)^2]
= (1/2)*(1.3225 + 1.8225)
= 1.5725

---

## 5. Backpropagation

We now compute gradients using the chain rule.

---

### Step 1: Output Layer Gradients

dL/do1 = (o1 - y1) = 1.15
dL/do2 = (o2 - y2) = 1.35

---

### Step 2: Gradients for Output Weights

dL/dv1 = dL/do1 * x = 1.15 * 3.5
dL/dv2 = dL/do1 * y = 1.15 * 7.5

dL/dv3 = dL/do2 * x = 1.35 * 3.5
dL/dv4 = dL/do2 * y = 1.35 * 7.5

---

### Step 3: Hidden Layer Error

This is the most important step.

The hidden neuron receives error from all outputs.

dL/dx = (dL/do1 * v1) + (dL/do2 * v3)
= (1.15 * 0.9) + (1.35 * 1.1)

dL/dy = (dL/do1 * v2) + (dL/do2 * v4)
= (1.15 * 1.0) + (1.35 * 1.2)

This shows:

* Error is not divided
* It is accumulated from all outputs

---

### Step 4: Gradients for Input Weights

For example:

dL/dw1 = dL/dx * a
dL/dw2 = dL/dx * b
dL/dw3 = dL/dx * c
dL/dw4 = dL/dx * d

Similarly for w5 to w8 using dL/dy

---

## 6. Weight Update Rule

Using gradient descent:

w_new = w_old - learning_rate * gradient

Example:

w1_new = w1 - lr * (dL/dw1)

---

## 7. Key Concepts

### 1. No Feature Splitting

Each hidden neuron sees all input features. There is no division of features. Each neuron learns a different combination.

### 2. Error Flow

Error flows backward from output to hidden to input weights.

### 3. Chain Rule

Every gradient is computed as a product of smaller derivatives.

### 4. Input vs Weights

Inputs (a, b, c, d) do not change. Only weights and biases are updated.

### 5. Hidden Layer Purpose

Hidden layers transform input into better representations for learning complex patterns.

---

## 8. Final Understanding

A neural network is a composition of functions:

f(g(h(x)))

Backpropagation applies the chain rule to compute gradients through each layer and update weights accordingly.

This is the core mechanism behind all deep learning models.
