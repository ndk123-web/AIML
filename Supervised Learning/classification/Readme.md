# Classification in Machine Learning

## Overview
Classification is a supervised learning technique where the model predicts discrete categorical labels (classes) rather than continuous values. This document covers the differences between Linear Regression, Binary Classification, and Multiple Classification.

---

## 1. Linear Regression

### What is it?
Linear Regression is used for **predicting continuous numerical values**. It's not technically a classification algorithm, but understanding it helps distinguish regression from classification.

### Formula
```
y = mx + b
```
Where:
- `y` = predicted output (continuous value)
- `m` = slope (coefficient)
- `x` = input feature
- `b` = y-intercept (bias)

For multiple features:
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

### Use Cases
- Predicting house prices
- Sales forecasting
- Temperature prediction
- Stock price estimation

### Output Range
- **-∞ to +∞** (any real number)

---

## 2. Binary Classification (Logistic Regression)

### What is it?
Binary Classification predicts one of **two possible classes** (0 or 1, Yes or No, True or False). It uses the **Sigmoid function** to convert continuous values into probabilities between 0 and 1.

### Sigmoid Function Formula
```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b` (linear combination)
- `e` = Euler's number (≈ 2.71828)
- Output range: **0 to 1** (probability)

### How it Works
1. Calculate linear combination: `z = wx + b`
2. Apply sigmoid function: `σ(z) = 1 / (1 + e^(-z))`
3. If `σ(z) ≥ 0.5` → Class 1 (Positive)
4. If `σ(z) < 0.5` → Class 0 (Negative)

### Sigmoid Function Properties
- **S-shaped curve**
- Maps any real number to range (0, 1)
- When z = 0, σ(z) = 0.5
- When z → +∞, σ(z) → 1
- When z → -∞, σ(z) → 0

### Use Cases
- Email spam detection (Spam/Not Spam)
- Medical diagnosis (Disease/No Disease)
- Customer churn prediction (Will Churn/Won't Churn)
- Loan approval (Approve/Reject)
- Insurance claim prediction (Buy/Won't Buy)

### Loss Function
**Binary Cross-Entropy (Log Loss)**:
```
Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

---

## 3. Multiple Classification (Multiclass Classification)

### What is it?
Multiple Classification predicts one class from **three or more possible classes**. It uses the **Softmax function** to convert raw scores into probabilities that sum to 1.

### Softmax Function Formula
```
Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```

Where:
- `zᵢ` = score for class i
- `e^(zᵢ)` = exponential of score i
- `Σⱼ e^(zⱼ)` = sum of exponentials of all class scores
- Output: Probability distribution across all classes

### How it Works
1. Calculate scores for each class: `z₁, z₂, z₃, ..., zₙ`
2. Apply exponential function to each score: `e^z₁, e^z₂, ..., e^zₙ`
3. Normalize by dividing each by the sum: `P(class_i) = e^(zᵢ) / Σ(e^(zⱼ))`
4. Select class with **highest probability**

### Example
Suppose we have 5 output classes with scores:
```
Classes:     A      B      C      D      E
Scores (z):  2.0    1.0    0.1    3.0    0.5

Step 1: Calculate exponentials
e^2.0 ≈ 7.39
e^1.0 ≈ 2.72
e^0.1 ≈ 1.11
e^3.0 ≈ 20.09
e^0.5 ≈ 1.65

Step 2: Sum all exponentials
Sum = 7.39 + 2.72 + 1.11 + 20.09 + 1.65 = 32.96

Step 3: Calculate probabilities
P(A) = 7.39/32.96 = 0.224 (22.4%)
P(B) = 2.72/32.96 = 0.083 (8.3%)
P(C) = 1.11/32.96 = 0.034 (3.4%)
P(D) = 20.09/32.96 = 0.609 (60.9%)  ← Highest
P(E) = 1.65/32.96 = 0.050 (5.0%)

Result: Class D is predicted (highest probability)
```

### Softmax Properties
- All probabilities are between 0 and 1
- Sum of all probabilities = 1
- Used in the **output layer** of neural networks
- Transforms raw scores into meaningful probabilities

### Use Cases
- Digit recognition (0-9): 10 classes
- Image classification (Cat/Dog/Bird/etc.)
- Language detection (English/Spanish/French/etc.)
- Product categorization
- Sentiment analysis (Positive/Neutral/Negative)
- Handwritten character recognition

### Loss Function
**Categorical Cross-Entropy**:
```
Loss = -Σᵢ yᵢ·log(ŷᵢ)
```

---

## Key Differences Summary

| Aspect | Linear Regression | Binary Classification | Multiple Classification |
|--------|------------------|----------------------|------------------------|
| **Type** | Regression | Classification | Classification |
| **Output** | Continuous value | 2 classes (0 or 1) | 3+ classes |
| **Activation** | None (Linear) | Sigmoid | Softmax |
| **Output Range** | -∞ to +∞ | 0 to 1 | Probability distribution (sum=1) |
| **Loss Function** | MSE (Mean Squared Error) | Binary Cross-Entropy | Categorical Cross-Entropy |
| **Example** | Predicting price | Spam detection | Digit recognition (0-9) |

---

## When to Use Which?

### Use Linear Regression when:
- You need to predict a **continuous numerical value**
- Example: price, temperature, age, salary

### Use Binary Classification when:
- You have **exactly 2 classes**
- Example: yes/no, pass/fail, positive/negative

### Use Multiple Classification when:
- You have **3 or more classes**
- Example: classifying images, categorizing text, multi-class sentiment

---

## Important Concepts

### 1. Why Sigmoid for Binary?
- Sigmoid squashes output to (0,1) range
- Can be interpreted as probability
- Natural extension of linear regression for classification

### 2. Why Softmax for Multiclass?
- Normalizes outputs into valid probability distribution
- All probabilities sum to 1
- Allows comparing relative confidence across classes

### 3. Decision Boundary
- **Binary**: Single threshold (usually 0.5)
- **Multiclass**: Multiple decision boundaries separating classes

---

## Additional Resources

- **Softmax Explanation Video**: https://www.youtube.com/watch?v=8ah-qhvaQqU
- Sigmoid vs Softmax: Softmax is generalization of sigmoid to multiple classes
- One-vs-Rest (OvR): Alternative approach for multiclass problems

---

## Code Examples

Check the subdirectories for practical implementations:
- `Binary Classification/` - Logistic regression examples with sigmoid
- `Multiple Classification/` - Multiclass problems using softmax  
