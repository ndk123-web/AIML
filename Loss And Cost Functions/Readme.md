# Loss and Cost Functions

## 1. Loss Functions

A loss function measures the error for a **single training example**. It quantifies how well the model's prediction matches the actual target value.

### Key Points:

- **Single Example**: Calculates error for one data point
- **Differentiable**: Must be differentiable to enable gradient-based optimization
- **Purpose**: Guides the model to improve predictions during training

## 2. Cost Functions

A cost function (also called objective function) calculates the **average error across all training examples** in a dataset.

### Key Points:

- **All Examples**: Averages losses from entire training set or batch
- **Optimization Target**: Minimized during training
- **Single Value**: Returns one scalar value representing overall model performance

---

## Detailed Example

### Data Setup:

- **Actual Values (y)**: [1, 2, 3, 4, 5]
- **Predicted Values (ŷ)**: [1.1, 2.2, 3.1, 3.9, 5.2]

### Individual Losses:

- **Loss 1**: |1 - 1.1| = 0.1
- **Loss 2**: |2 - 2.2| = 0.2
- **Loss 3**: |3 - 3.1| = 0.1
- **Loss 4**: |4 - 3.9| = 0.1
- **Loss 5**: |5 - 5.2| = 0.2

### Cost Calculation:

**Total Cost** = (0.1 + 0.2 + 0.1 + 0.1 + 0.2) / 5 = **0.14**

---

## Common Cost Functions

### 1. Mean Squared Error (MSE)

**Formula**: (1/n) \* Σ(y - ŷ)²

**Characteristics**:

- **Penalizes Large Errors**: Squares amplify large differences
- **Differentiable**: Smooth gradient for optimization
- **Common Use**: Regression problems
- **Range**: [0, ∞)

**Example Calculation**:

- MSE = (0.1² + 0.2² + 0.1² + 0.1² + 0.2²) / 5
- MSE = (0.01 + 0.04 + 0.01 + 0.01 + 0.04) / 5 = 0.022

### 2. Mean Absolute Error (MAE)

**Formula**: (1/n) \* Σ|y - ŷ|

**Characteristics**:

- **Linear Penalty**: Treats all errors proportionally
- **Robust**: Less sensitive to outliers than MSE
- **Interpretation**: Same units as target variable
- **Range**: [0, ∞)

**Example Calculation**:

- MAE = (0.1 + 0.2 + 0.1 + 0.1 + 0.2) / 5 = 0.14

### 3. Root Mean Squared Error (RMSE)

**Formula**: √((1/n) \* Σ(y - ŷ)²)

**Characteristics**:

- **Interpretable**: Same units as target variable
- **Sensitive to Outliers**: Like MSE but with unit consistency
- **Common Use**: Regression, forecasting
- **Range**: [0, ∞)

**Example Calculation**:

- RMSE = √0.022 ≈ 0.148

---

## Classification Loss Functions

### 1. Binary Cross-Entropy

**Use Case**: Binary classification (2 classes)
**Formula**: -[y·log(ŷ) + (1-y)·log(1-ŷ)]

### 2. Categorical Cross-Entropy

**Use Case**: Multi-class classification (one-hot encoded)
**Formula**: -Σ(y·log(ŷ))

### 3. Sparse Categorical Cross-Entropy

**Use Case**: Multi-class classification (integer labels)
**Formula**: -log(ŷ[class_index])

---

## Training Process

### Epoch

One complete pass through the entire training dataset:

1. **Forward Pass**: Calculate predictions
2. **Loss Calculation**: Compute individual losses
3. **Cost Calculation**: Average all losses
4. **Backward Pass**: Update weights using gradients
5. **Repeat**: For each batch in the dataset

### Key Concepts:

- **Batch Size**: Number of examples processed before updating weights
- **Iterations**: Number of batches needed to complete one epoch
- **Learning Rate**: Step size for weight updates
- **Convergence**: When cost stops decreasing significantly

---

## Choosing the Right Loss Function

### For Regression:

- **MSE**: When large errors should be heavily penalized
- **MAE**: When outliers should have less influence
- **RMSE**: When interpretable units are needed

### For Classification:

- **Binary Cross-Entropy**: Binary (0/1) problems
- **Categorical Cross-Entropy**: Multi-class with one-hot encoding
- **Sparse Categorical Cross-Entropy**: Multi-class with integer labels

### Key Considerations:

- **Problem Type**: Regression vs Classification
- **Data Distribution**: Presence of outliers
- **Output Format**: Probabilities vs discrete values
- **Interpretability**: Unit consistency matters

### Log

- Log is important
- some properites of log:
  - log base x of some y where y always greater than equal to 0
  - loge (natural log) is commonly used in machine learning
  - log10 (common log) is commonly used in engineering
  - log0 is undefined
  - log1 is always 0

  - loge (0.01) = -4.605
  - loge (0.1) = -2.303
  - loge (0.5) = -0.693
  - loge (1) = 0
  - loge (2) = 0.693
  - loge (10) = 2.303
  - loge (100) = 4.605

  - as u can see when it is grater than 0 then log is increasing
  - as u can see when it is less than 0 then log is decreasing
  - as u can see when it is equal to 0 then log is 0
