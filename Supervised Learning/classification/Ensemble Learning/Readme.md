## Ensemble Learning

Ensemble learning ka matlab hai: multiple models ko combine karke ek powerful predictive model banana.

**Core idea**: Jab multiple models ki predictions combine karte hain, to final result akeli model se zyada accurate aur robust hota hai.

Yeh "wisdom of the crowd" concept pe based hai - ek expert se zyada, multiple experts ki combined opinion aksar better hoti hai.

---

## 1. Why Ensemble Works?

- **Error Reduction**: different models different types ki errors karti hain. Jab combine karte hain to errors cancel out ho jati hain.
- **Variance Reduction**: unstable models ko combine karke stable predictions mil sakti hain.
- **Bias Reduction**: weak learners ko combine karke strong learner ban sakta hai.
- **Robustness**: single model ka performance kharab bhi ho to ensemble overall achha perform kar sakta hai.

---

## 2. Main Types of Ensemble Methods

### A. Bagging (Bootstrap Aggregating)

**Concept**:
- Training data se multiple random subsets (with replacement) liye jate hain.
- Har subset pe alag model train hota hai.
- Final prediction sab models ki **voting** (classification) ya **averaging** (regression) se aati hai.

**Key Points**:
- Parallel training - saari models independently train hoti hain.
- **Variance reduction** main goal hai.
- High variance wale models (like decision trees) ke liye best.

**Example Algorithm**: Random Forest

---

### B. Boosting

**Concept**:
- Models sequentially train hoti hain.
- Har naya model **previous model ki errors** ko focus karke seekhta hai.
- Misclassified points ko higher weight milta hai.
- Final prediction weighted combination hota hai.

**Key Points**:
- Sequential training - models ek ke baad ek train hoti hain.
- **Bias reduction** focus hai.
- Weak learners ko strong learner bana sakta hai.

**Example Algorithms**: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

---

### C. Stacking (Stacked Generalization)

**Concept**:
- Multiple diverse base models (level-0) train karte hain.
- Base models ki predictions ko input ke taur pe use karke ek **meta-model** (level-1) train hota hai.
- Meta-model final prediction deta hai.

**Key Points**:
- Different types ki models combine kar sakte hain (e.g., SVM + Random Forest + Logistic Regression).
- Flexible lekin computationally expensive.

---

## 3. Popular Ensemble Algorithms (Detailed)

### 3.1 Random Forest (Bagging-based)

**How it works**:
- Multiple decision trees parallel banate hain.
- Har tree ko:
  - Random subsample of data milta hai (bootstrapping).
  - Random subset of features milti hai har split pe.
- Final prediction: majority voting (classification) ya mean (regression).

**Why Random Forest Better Than Single Tree**:
- Single decision tree overfit kar sakta hai.
- Random Forest averaging ki wajah se generalization better hoti hai.

**Hyperparameters**:
- `n_estimators`: kitne trees (default 100, try 100-500)
- `max_depth`: tree ka maximum depth (control overfitting)
- `max_features`: har split pe kitne features consider karne (sqrt, log2, ya int)
- `min_samples_split`: node split ke liye minimum samples
- `min_samples_leaf`: leaf node me minimum samples

**Code Example**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features='sqrt',
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

# Feature importance
import pandas as pd
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_imp)
```

---

### 3.2 AdaBoost (Adaptive Boosting)

**How it works**:
1. Pehle weak learner train hota hai (usually shallow decision tree - stump).
2. Misclassified samples ka weight badha diya jata hai.
3. Next model in weighted samples pe focus karta hai.
4. Process repeat hoti hai.
5. Final prediction weighted sum of all weak learners.

**Key Insight**:
- Weak learners (slightly better than random guessing) ko combine karke strong learner banta hai.
- Sequential training hoti hai.

**Hyperparameters**:
- `n_estimators`: kitne weak learners
- `learning_rate`: har learner ka contribution (low = slow learning but better generalization)
- `base_estimator`: weak learner type (default: DecisionTreeClassifier with max_depth=1)

**Code Example**:
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
```

---

### 3.3 Gradient Boosting

**How it works**:
1. Initial prediction (usually mean/median).
2. Residuals (actual - predicted) calculate karo.
3. Next model residuals ko predict karne ke liye train hota hai.
4. Prediction update hota hai: `prediction_new = prediction_old + learning_rate * residual_prediction`.
5. Steps repeat hote hain.

**Key Difference from AdaBoost**:
- AdaBoost weights samples, Gradient Boosting fits to residuals.
- More flexible loss functions use kar sakta hai.

**Hyperparameters**:
- `n_estimators`: number of boosting stages
- `learning_rate`: shrinks contribution of each tree (0.01 to 0.3 common)
- `max_depth`: individual tree depth (3-5 common)
- `subsample`: fraction of samples for each tree (< 1.0 -> stochastic GB)

**Code Example**:
```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
```

---

### 3.4 XGBoost (Extreme Gradient Boosting)

**Why XGBoost Special**:
- Gradient Boosting ka optimized aur fast implementation.
- Regularization built-in (L1 and L2).
- Parallel processing support.
- Handles missing values automatically.
- Tree pruning (depth-first approach).

**Installation**:
```bash
pip install xgboost
```

**Key Hyperparameters**:
- `n_estimators`: boosting rounds
- `learning_rate` (or `eta`): step size shrinkage (0.01-0.3)
- `max_depth`: tree depth (3-10)
- `subsample`: row sampling ratio (0.5-1.0)
- `colsample_bytree`: column sampling ratio (0.5-1.0)
- `gamma`: minimum loss reduction for split (regularization)
- `reg_alpha`: L1 regularization
- `reg_lambda`: L2 regularization

**Code Example**:
```python
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

# Feature importance plot
xgb.plot_importance(xgb_clf)
```

---

### 3.5 LightGBM (Light Gradient Boosting Machine)

**Why LightGBM**:
- XGBoost se bhi fast (especially large datasets pe).
- **Leaf-wise** tree growth (vs level-wise in XGBoost) -> deeper, more accurate trees.
- Memory efficient.
- Categorical features directly handle kar sakta hai.

**Installation**:
```bash
pip install lightgbm
```

**Code Example**:
```python
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,  # -1 means no limit
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_clf.fit(X_train, y_train)
y_pred = lgb_clf.predict(X_test)
```

---

### 3.6 CatBoost (Categorical Boosting)

**Why CatBoost**:
- Categorical features ko specially handle karta hai (encoding automatically).
- Overfitting resistant.
- Less hyperparameter tuning required.
- Good default parameters.

**Installation**:
```bash
pip install catboost
```

**Code Example**:
```python
from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=False,
    random_state=42
)

# Agar categorical features hain
cat_features = [0, 2, 5]  # column indices
cat_clf.fit(X_train, y_train, cat_features=cat_features)
y_pred = cat_clf.predict(X_test)
```

---

### 3.7 Voting Classifier (Simple Ensemble)

**How it works**:
- Multiple diverse models train karte hain.
- Prediction time pe voting hoti hai:
  - **Hard voting**: majority class vote.
  - **Soft voting**: average probabilities (works better when models output probabilities).

**Code Example**:
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('svc', SVC(probability=True))
    ],
    voting='soft'  # or 'hard'
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
```

---

## 4. Bagging vs Boosting (Quick Comparison)

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Variance reduction | Bias reduction |
| Data sampling | Random subsets with replacement | Weighted sampling focusing on errors |
| Models | Independent | Dependent on previous models |
| Best for | High variance models (e.g., deep trees) | Weak learners |
| Example | Random Forest | AdaBoost, XGBoost, Gradient Boosting |
| Overfitting risk | Lower | Higher (if not tuned properly) |

---

## 5. When to Use Which Ensemble?

- **Random Forest**:
  - General purpose classification/regression
  - Feature importance chahiye
  - Less tuning, robust performance
  - Medium to large datasets

- **AdaBoost**:
  - Simple weak learners ko combine karna hai
  - Relatively small datasets
  - Less sensitive to overfitting than other boosting methods

- **Gradient Boosting / XGBoost**:
  - Competition-winning performance chahiye
  - Hyperparameter tuning kar sakte ho
  - Structured/tabular data
  - XGBoost jab speed + performance dono chahiye

- **LightGBM**:
  - Very large datasets
  - Speed critical priority
  - Deep trees chahiye

- **CatBoost**:
  - Lots of categorical features
  - Less tuning time available
  - Good default performance chahiye

- **Voting/Stacking**:
  - Multiple diverse models already trained hain
  - Squeeze out last bit of performance

---

## 6. Common Hyperparameter Tuning Tips

### Random Forest:
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, random_state=42)
search.fit(X_train, y_train)
print(search.best_params_)
```

### XGBoost:
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_clf = xgb.XGBClassifier(random_state=42)
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy')
search.fit(X_train, y_train)
```

---

## 7. Best Practices

1. **Start Simple**: Pehle Random Forest try karo as baseline.

2. **Feature Engineering**: Ensemble methods powerful hain lekin good features se performance aur bhi better hoti hai.

3. **Cross-Validation**: Hamesha CV se evaluate karo, single train-test split pe rely mat karo.

4. **Avoid Overfitting in Boosting**:
   - Learning rate kam rakho (0.01-0.1).
   - Early stopping use karo (validation set pe).
   - Regularization parameters tune karo.

5. **Feature Importance**: Random Forest ya XGBoost se feature importance nikalo, unnecessary features remove karo.

6. **Imbalanced Data**: `class_weight='balanced'` ya SMOTE jaise techniques use karo.

7. **Computational Cost**: Boosting methods training me slow ho sakte hain. Large datasets pe LightGBM consider karo.

8. **Ensemble of Ensembles**: XGBoost + LightGBM + CatBoost ko voting/stacking se combine karke aur better results mil sakte hain (Kaggle competitions me common).

---

## 8. Stacking Example (Advanced)

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Base models (level-0)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

# Meta-model (level-1)
meta_model = LogisticRegression()

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # cross-validation for base model predictions
)

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred)}")
```

---

## 9. Key Takeaways

- **Ensemble = Multiple models ka combination = Better performance**.
- **Bagging** (Random Forest) -> Variance kam karta hai, parallel training.
- **Boosting** (AdaBoost, XGBoost, LightGBM, CatBoost) -> Bias kam karta hai, sequential training, errors pe focus.
- **Stacking** -> Different models ko meta-learner se combine karta hai.
- Ensemble methods competition-winning performance dete hain lekin interpretability kam hoti hai compared to single models.
- Hamesha cross-validation karo aur hyperparameter tuning karo for best results.

---

## 10. One-Line Summary

Ensemble learning multiple models ko intelligent tarike se combine karke single model se kaafi better predictive accuracy achieve karta hai, jisme bagging variance reduce karta hai aur boosting bias reduce karta hai.
