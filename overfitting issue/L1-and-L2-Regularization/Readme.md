# L1 and L2 Regularization

## Overview

Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. This penalty discourages the model from learning overly complex patterns that might not generalize well to new data.

## L1 Regularization (Lasso Regression)

### What is L1?
- **L1** stands for **Lasso** (Least Absolute Shrinkage and Selection Operator)
- Adds the **absolute value** of the magnitude of coefficients as a penalty term to the loss function
- Performs both regularization and feature selection

### Mathematical Formula
```
Cost Function = Loss Function + λ × Σ|w|
```
Where:
- `λ` (lambda) is the regularization parameter
- `w` represents the model weights/coefficients
- `Σ|w|` is the sum of absolute values of all weights

### Key Characteristics
1. **Sparse Solutions**: L1 can shrink some coefficients to **exactly zero**
2. **Feature Selection**: Automatically performs feature selection by eliminating irrelevant features
3. **Model Interpretability**: Produces simpler, more interpretable models
4. **Works well** when you have many features and suspect only a few are important

### When to Use L1
- When you have many features and want automatic feature selection
- When you need a simpler, more interpretable model
- When you believe only a few features are truly important
- When dealing with high-dimensional data

---

## L2 Regularization (Ridge Regression)

### What is L2?
- **L2** stands for **Ridge Regression**
- Adds the **squared magnitude** of coefficients as a penalty term to the loss function
- Shrinks coefficients but doesn't eliminate them completely

### Mathematical Formula
```
Cost Function = Loss Function + λ × Σ(w²)
```
Where:
- `λ` (lambda) is the regularization parameter
- `w` represents the model weights/coefficients
- `Σ(w²)` is the sum of squared values of all weights

### Key Characteristics
1. **Shrinks Coefficients**: Reduces the magnitude of all coefficients towards zero (but not to zero)
2. **No Feature Selection**: All features remain in the model
3. **Handles Multicollinearity**: Works well when features are highly correlated
4. **Smooth Penalty**: The squared term makes it differentiable everywhere

### When to Use L2
- When all features are potentially important
- When features are highly correlated (multicollinearity)
- When you want to keep all features but reduce their impact
- When you need stable, consistent predictions

---

## L1 vs L2: Key Differences

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Penalty Term** | Sum of absolute values: Σ\|w\| | Sum of squares: Σ(w²) |
| **Feature Selection** | Yes - sets some weights to 0 | No - shrinks all weights |
| **Sparsity** | Produces sparse models | Produces dense models |
| **Computation** | Not differentiable at zero | Differentiable everywhere |
| **Multicollinearity** | May randomly select one feature | Handles well, distributes weights |
| **Use Case** | Feature selection needed | All features important |

---

## Elastic Net (Combination of L1 and L2)

### What is Elastic Net?
- Combines both L1 and L2 regularization
- Gets benefits of both Lasso and Ridge

### Mathematical Formula
```
Cost Function = Loss Function + λ₁ × Σ|w| + λ₂ × Σ(w²)
```

### When to Use
- When you want feature selection (L1) but also need to handle correlated features (L2)
- When you have groups of correlated features
- Best of both worlds approach

---

## Hyperparameter: λ (Lambda)

### What is λ?
- Controls the strength of regularization
- Higher λ = stronger regularization (simpler model, higher bias)
- Lower λ = weaker regularization (more complex model, higher variance)

### Effects of Different λ Values

| λ Value | Effect |
|---------|--------|
| **λ = 0** | No regularization (standard regression) |
| **Small λ** | Slight regularization, model close to unregularized |
| **Medium λ** | Balanced regularization |
| **Large λ** | Strong regularization, may underfit |

### Finding Optimal λ
- Use **Cross-Validation** to find the best λ value
- Techniques: GridSearchCV, RandomizedSearchCV
- Plot validation error vs λ to visualize

---

## Practical Implementation in Python

### Using Scikit-learn

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score

# L1 Regularization (Lasso)
lasso = Lasso(alpha=0.1)  # alpha is λ
lasso.fit(X_train, y_train)

# L2 Regularization (Ridge)
ridge = Ridge(alpha=1.0)  # alpha is λ
ridge.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: mix of L1 and L2
elastic.fit(X_train, y_train)
```

### Finding Optimal Alpha (λ)

```python
from sklearn.model_selection import GridSearchCV

# For Ridge
params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(Ridge(), params, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.best_params_['alpha']}")

# For Lasso
lasso_cv = GridSearchCV(Lasso(), params, cv=5)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha: {lasso_cv.best_params_['alpha']}")
```

---

## Advantages and Disadvantages

### L1 (Lasso)
**Advantages:**
- Automatic feature selection
- Produces sparse, interpretable models
- Good for high-dimensional data

**Disadvantages:**
- Can be unstable with correlated features
- May arbitrarily select one feature from a group of correlated features
- Not differentiable at zero (optimization challenges)

### L2 (Ridge)
**Advantages:**
- Handles multicollinearity well
- Stable and smooth optimization
- Usually better predictive performance when all features are relevant

**Disadvantages:**
- No feature selection
- Less interpretable with many features
- All features remain in the model

---

## Summary

- **Use L1 (Lasso)** when you need feature selection and a sparse model
- **Use L2 (Ridge)** when all features are important and you have multicollinearity
- **Use Elastic Net** when you want both feature selection and handling of correlated features
- Always use **cross-validation** to find the optimal regularization parameter (λ/alpha)
- Regularization helps prevent overfitting and improves model generalization 