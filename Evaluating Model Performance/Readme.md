## Evaluating Model Performance

Model evaluation is critical to understand how well your machine learning model will perform on unseen data. Simply training a model is not enough - you need to measure its effectiveness using appropriate metrics and validation strategies.

---

## 1. Training Strategies

### 1.1 Using 100% Data for Training

**Approach**: Train the model on the entire dataset.

**Pros**:
- Model learns from all available data.
- Maximum information utilization.

**Cons**:
- **No way to evaluate generalization** - you can't test on unseen data.
- High risk of overfitting without knowing it.
- No validation strategy.

**When to use**: 
- Only when you have a separate, independent test dataset.
- After final model selection and tuning (training final production model).

---

### 1.2 Train-Test Split

**Approach**: Split dataset into training set (typically 70-80%) and test set (20-30%).

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # for reproducibility
    stratify=y          # maintain class distribution
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Pros**:
- Simple and fast.
- Clear separation between training and evaluation.
- Good for large datasets.

**Cons**:
- **Data imbalance issue**: If classes are imbalanced, random split might not maintain proportions.
- Performance depends on which samples end up in train vs test.
- Less training data available (some data reserved for testing).
- Single evaluation - might not be representative.

**Best Practice**: Use `stratify=y` parameter to maintain class distribution in both sets.

---

### 1.3 K-Fold Cross-Validation

**Approach**: 
1. Split data into K equal folds (typically K=5 or K=10).
2. Train K different models, each time using K-1 folds for training and 1 fold for validation.
3. Average the performance across all K folds.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")
```

**Pros**:
- More reliable performance estimate.
- All data used for both training and validation.
- Reduces variance in performance estimates.
- Better for smaller datasets.

**Cons**:
- Computationally expensive (trains K models).
- Takes more time than simple train-test split.

**Variants**:
- **Stratified K-Fold**: Maintains class distribution in each fold (recommended for classification).
- **Leave-One-Out (LOO)**: K = number of samples (extreme case, very expensive).
- **Time Series Split**: For time-series data (no random shuffling).

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(
    model, X, y, 
    cv=skf,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

print(f"Test Accuracy: {results['test_accuracy'].mean():.3f}")
print(f"Test F1-Score: {results['test_f1'].mean():.3f}")
```

---

## 2. Classification Metrics

### 2.1 Confusion Matrix

Foundation of most classification metrics. Shows true positives, false positives, true negatives, and false negatives.

```
                    Predicted
                 Positive  Negative
Actual Positive     TP        FN
       Negative     FP        TN
```

- **True Positive (TP)**: Correctly predicted positive class.
- **True Negative (TN)**: Correctly predicted negative class.
- **False Positive (FP)**: Incorrectly predicted positive (Type I error).
- **False Negative (FN)**: Incorrectly predicted negative (Type II error).

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
```

---

### 2.2 Accuracy

**Definition**: Ratio of correct predictions to total predictions.

**Formula**: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

**When to use**: Balanced datasets where all classes are equally important.

**Limitation**: Misleading for imbalanced datasets.

**Example**: In a dataset with 95% negative and 5% positive samples, a model that always predicts negative gets 95% accuracy but is useless.

---

### 2.3 Precision

**Definition**: Of all positive predictions, how many were actually positive?

**Formula**: `Precision = TP / (TP + FP)`

**Use case**: When false positives are costly.

**Example**: Spam detection - you don't want legitimate emails marked as spam.

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.3f}")
```

---

### 2.4 Recall (Sensitivity / True Positive Rate)

**Definition**: Of all actual positive samples, how many did we correctly identify?

**Formula**: `Recall = TP / (TP + FN)`

**Use case**: When false negatives are costly.

**Example**: Disease detection - you don't want to miss positive cases.

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.3f}")
```

---

### 2.5 F1-Score

**Definition**: Harmonic mean of precision and recall. Balances both metrics.

**Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Use case**: When you need balance between precision and recall, especially on imbalanced datasets.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3f}")
```

**Variants**:
- **F-beta Score**: Weighted version that can emphasize precision or recall more.

---

### 2.6 Classification Report

Comprehensive report showing precision, recall, F1-score, and support for each class.

```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)
```

**Output example**:
```
              precision    recall  f1-score   support

     Class 0       0.85      0.92      0.88       100
     Class 1       0.90      0.82      0.86        90

    accuracy                           0.87       190
   macro avg       0.88      0.87      0.87       190
weighted avg       0.87      0.87      0.87       190
```

---

### 2.7 ROC Curve and AUC

**ROC (Receiver Operating Characteristic) Curve**: 
- Plots True Positive Rate (Recall) vs False Positive Rate at various threshold values.
- Shows trade-off between sensitivity and specificity.

**AUC (Area Under the Curve)**:
- Single number summary of ROC curve.
- Range: 0 to 1 (0.5 = random guessing, 1.0 = perfect).
- Higher is better.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calculate AUC
auc = roc_auc_score(y_test, y_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**Use case**: 
- Comparing multiple models.
- Evaluating model across different thresholds.
- Works well for imbalanced datasets.

---

### 2.8 Precision-Recall Curve

Alternative to ROC curve, especially useful for **highly imbalanced datasets**.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'AP = {ap:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

---

### 2.9 Log Loss (Cross-Entropy Loss)

Measures the performance of classification models that output probabilities.

**Formula**: Penalizes confident wrong predictions heavily.

```python
from sklearn.metrics import log_loss

logloss = log_loss(y_test, y_proba)
print(f"Log Loss: {logloss:.3f}")
```

**Use case**: When probability calibration is important (e.g., risk assessment).

---

## 3. Regression Metrics

### 3.1 Mean Absolute Error (MAE)

**Definition**: Average absolute difference between predictions and actual values.

**Formula**: `MAE = (1/n) × Σ|y_actual - y_pred|`

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.3f}")
```

**Pros**: Easy to interpret, same unit as target variable.

**Cons**: Doesn't penalize large errors heavily.

---

### 3.2 Mean Squared Error (MSE)

**Definition**: Average of squared differences.

**Formula**: `MSE = (1/n) × Σ(y_actual - y_pred)²`

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.3f}")
```

**Pros**: Penalizes large errors more than MAE.

**Cons**: Unit is squared, harder to interpret.

---

### 3.3 Root Mean Squared Error (RMSE)

**Definition**: Square root of MSE.

**Formula**: `RMSE = √MSE`

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.3f}")
```

**Pros**: Same unit as target variable, penalizes large errors.

---

### 3.4 R² Score (Coefficient of Determination)

**Definition**: Proportion of variance in the dependent variable explained by the model.

**Formula**: `R² = 1 - (SS_residual / SS_total)`

**Range**: -∞ to 1
- 1.0 = Perfect predictions
- 0.0 = Model performs as well as mean baseline
- Negative = Model worse than mean baseline

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.3f}")
```

**Most commonly used metric for regression.**

---

### 3.5 Mean Absolute Percentage Error (MAPE)

**Definition**: Average percentage difference between predictions and actual values.

**Formula**: `MAPE = (100/n) × Σ|(y_actual - y_pred) / y_actual|`

```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape:.2f}%")
```

**Pros**: Scale-independent, easy to interpret.

**Cons**: Undefined when actual value is 0.

---

## 4. Avoiding Common Pitfalls

### 4.1 Overfitting

**Definition**: Model performs well on training data but poorly on test data.

**Symptoms**:
- High training accuracy, low test accuracy.
- Large gap between training and validation scores.

**Solutions**:
- Use regularization (L1, L2).
- Reduce model complexity.
- Increase training data.
- Use cross-validation.
- Early stopping (for iterative algorithms).
- Dropout (for neural networks).

```python
# Check for overfitting
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

---

### 4.2 Underfitting

**Definition**: Model is too simple to capture patterns in the data.

**Symptoms**:
- Low training accuracy.
- Low test accuracy.
- High bias.

**Solutions**:
- Increase model complexity.
- Add more features.
- Reduce regularization.
- Train longer.

---

### 4.3 Data Leakage

**Definition**: Information from test set leaks into training process.

**Common causes**:
- Scaling before splitting (fit scaler on entire dataset).
- Using future information in time-series.
- Duplicate samples across train and test.

**Prevention**:
```python
# WRONG - leakage
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on all data
X_train, X_test = train_test_split(X_scaled, ...)

# CORRECT - no leakage
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
X_test_scaled = scaler.transform(X_test)  # Transform test
```

---

### 4.4 Class Imbalance

**Problem**: When one class has significantly more samples than others.

**Solutions**:

1. **Use stratified splitting**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2
)
```

2. **Class weights**:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
```

3. **Resampling**:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversampling minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

4. **Use appropriate metrics**: Focus on precision, recall, F1-score, AUC instead of just accuracy.

---

## 5. Cross-Validation Strategies

### 5.1 Standard K-Fold

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
```

---

### 5.2 Stratified K-Fold (Recommended for Classification)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

---

### 5.3 Time Series Split (For Temporal Data)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Score: {score:.3f}")
```

---

### 5.4 Leave-One-Out Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
print(f"Mean accuracy: {scores.mean():.3f}")
```

**Warning**: Very expensive for large datasets (trains n models where n = number of samples).

---

## 6. Hyperparameter Tuning with Cross-Validation

### 6.1 Grid Search CV

Exhaustively searches through specified parameter combinations.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
```

---

### 6.2 Randomized Search CV

Randomly samples parameter combinations (faster than Grid Search).

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,  # number of parameter settings sampled
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
```

---

## 7. Model Comparison Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True),
    'XGBoost': xgb.XGBClassifier()
}

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['mean'])
print(f"\nBest model: {best_model_name}")
```

---

## 8. Best Practices

1. **Always use cross-validation** for reliable performance estimates, especially on smaller datasets.

2. **Use stratified splits** for classification to maintain class distribution.

3. **Choose metrics appropriate to your problem**:
   - Imbalanced data: Precision, Recall, F1, AUC
   - Balanced data: Accuracy
   - Probability calibration important: Log Loss
   - Regression: R², RMSE, MAE

4. **Prevent data leakage**: Always fit preprocessing (scaling, encoding) only on training data.

5. **Check for overfitting**: Monitor both training and validation performance.

6. **Use multiple metrics**: Don't rely on a single metric.

7. **Set random seeds** for reproducibility.

8. **Document your evaluation strategy** so results can be replicated.

9. **Use learning curves** to diagnose bias-variance issues.

10. **Final test set**: Keep a completely separate test set that is NEVER used during model selection or tuning.

---

## 9. Complete Evaluation Pipeline Example

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# 1. Split data (hold out final test set)
X_temp, X_final_test, y_temp, y_final_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

# 2. Scale features (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_final_test_scaled = scaler.transform(X_final_test)

# 3. Hyperparameter tuning with cross-validation
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# 4. Evaluate on validation set
y_val_pred = best_model.predict(X_val_scaled)
y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]

print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred))
print(f"AUC-ROC: {roc_auc_score(y_val, y_val_proba):.3f}")

# 5. Final evaluation on held-out test set
y_final_pred = best_model.predict(X_final_test_scaled)
y_final_proba = best_model.predict_proba(X_final_test_scaled)[:, 1]

print("\nFinal Test Set Performance:")
print(classification_report(y_final_test, y_final_pred))
print(f"AUC-ROC: {roc_auc_score(y_final_test, y_final_proba):.3f}")

# 6. Confusion matrix
cm = confusion_matrix(y_final_test, y_final_pred)
print(f"\nConfusion Matrix:\n{cm}")
```

---

## 10. Summary

- **Training strategies**: Train-test split for quick evaluation, K-fold CV for reliable estimates.
- **Classification metrics**: Accuracy (balanced data), Precision/Recall/F1 (imbalanced data), AUC-ROC (threshold-independent).
- **Regression metrics**: R² (variance explained), RMSE/MAE (prediction error).
- **Avoid pitfalls**: Overfitting, underfitting, data leakage, class imbalance.
- **Cross-validation**: Essential for model selection and hyperparameter tuning.
- **Always keep a final held-out test set** for unbiased performance estimation.

Proper model evaluation ensures your model will perform well on real-world, unseen data rather than just memorizing training examples.
