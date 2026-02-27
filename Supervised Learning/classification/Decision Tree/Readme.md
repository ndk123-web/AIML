# Decision Tree

## Overview
A **Decision Tree** is a supervised learning algorithm that builds a tree-like model of decisions. It splits the data into subsets based on feature values, creating a flowchart structure where each internal node represents a "test" on a feature, each branch represents the outcome, and each leaf node represents a class label (classification) or value (regression).

---

## How Decision Trees Work

### Basic Concept
1. **Start at Root Node**: Contains all training data
2. **Split the Data**: Choose the best feature to split on
3. **Create Branches**: Each branch represents a possible value or range
4. **Repeat Recursively**: For each subset, repeat the splitting process
5. **Stop at Leaf Nodes**: When a stopping criterion is met (pure node, max depth, etc.)

### Decision Process
Each node asks a **Yes/No question** about a feature:
- "Is Age > 30?"
- "Is Salary > 50000?"
- "Is City == 'Mumbai'?"

Based on the answer, we move down the appropriate branch until we reach a leaf node with the prediction.

---

## Tree Splitting Criteria

The key challenge: **How to choose the best feature and split point?**

Decision trees aim to create splits that result in **maximum purity** (homogeneous subsets). We use mathematical criteria to measure impurity and choose splits that **minimize impurity** (or maximize information gain).

### Two Main Methods:

---

## 1. Gini Impurity (Gini Index)

### What is it?
Gini Impurity measures the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the distribution of labels in the subset.

### Formula
```
Gini = 1 - Σᵢ (pᵢ)²
```

For **binary classification** (2 classes):
```
Gini = 1 - (p₁² + p₂²)
```

Where:
- `p₁` = proportion of class 1 = (count of class 1) / total
- `p₂` = proportion of class 2 = (count of class 2) / total
- Gini ranges from **0 to 0.5**
  - **0** = Perfect purity (all samples belong to one class)
  - **0.5** = Maximum impurity (equal distribution)

### Example Calculation

**Scenario**: We have 10 samples to classify (Buy Insurance: Yes/No)
- 7 samples: Yes (will buy)
- 3 samples: No (won't buy)

```
Step 1: Calculate proportions
p₁ = 7/10 = 0.7
p₂ = 3/10 = 0.3

Step 2: Apply Gini formula
Gini = 1 - (p₁² + p₂²)
     = 1 - (0.7² + 0.3²)
     = 1 - (0.49 + 0.09)
     = 1 - 0.58
     = 0.42
```

**Gini = 0.42** (indicates impurity - mix of both classes)

### Perfect Purity Example
- 10 samples: All Yes
- p₁ = 10/10 = 1.0, p₂ = 0/10 = 0.0
- Gini = 1 - (1² + 0²) = 1 - 1 = **0** ✅ Perfect!

### Maximum Impurity Example
- 5 samples: Yes, 5 samples: No
- p₁ = 0.5, p₂ = 0.5
- Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = **0.5** (worst case)

---

## 2. Entropy (Information Gain)

### What is it?
Entropy measures the **disorder** or **uncertainty** in the data. Higher entropy means more disorder. Information Gain measures the **reduction in entropy** after a split.

### Formula
```
Entropy = -Σᵢ (pᵢ · log₂(pᵢ))
```

For **binary classification**:
```
Entropy = -(p₁ · log₂(p₁) + p₂ · log₂(p₂))
```

Where:
- `p₁`, `p₂` = proportions of each class
- `log₂` = logarithm base 2
- Entropy ranges from **0 to 1** (for binary)
  - **0** = Perfect purity (no disorder)
  - **1** = Maximum impurity (maximum disorder)

### Example Calculation

**Same scenario**: 7 Yes, 3 No out of 10 samples

```
Step 1: Calculate proportions
p₁ = 7/10 = 0.7
p₂ = 3/10 = 0.3

Step 2: Apply Entropy formula
Entropy = -(p₁ · log₂(p₁) + p₂ · log₂(p₂))
        = -(0.7 · log₂(0.7) + 0.3 · log₂(0.3))
        = -(0.7 · (-0.515) + 0.3 · (-1.737))
        = -(-0.360 + (-0.521))
        = -(-0.881)
        = 0.881
```

**Entropy = 0.881** (high disorder)

### Perfect Purity Example
- All samples belong to one class
- p₁ = 1.0, p₂ = 0.0
- Entropy = -(1 · log₂(1) + 0 · log₂(0)) = 0 ✅ Perfect!

Note: By convention, 0 · log₂(0) = 0

### Maximum Impurity Example
- Equal distribution: p₁ = 0.5, p₂ = 0.5
- Entropy = -(0.5 · log₂(0.5) + 0.5 · log₂(0.5))
- Entropy = -(0.5 · (-1) + 0.5 · (-1)) = 1.0 (maximum)

---

## Information Gain

### What is it?
**Information Gain** measures how much information (reduction in entropy) we get by splitting on a particular feature.

### Formula
```
Information Gain = Entropy(parent) - Weighted Average Entropy(children)
```

### Example
**Parent Node**: 10 samples (6 Yes, 4 No)
```
Entropy(parent) = -(0.6 · log₂(0.6) + 0.4 · log₂(0.4))
                = 0.971
```

**Split on "Age > 30"**:
- **Left Child** (Age ≤ 30): 6 samples (2 Yes, 4 No)
  - Entropy = -(2/6 · log₂(2/6) + 4/6 · log₂(4/6)) = 0.918
  
- **Right Child** (Age > 30): 4 samples (4 Yes, 0 No)
  - Entropy = 0 (pure!)

```
Weighted Entropy = (6/10) · 0.918 + (4/10) · 0 = 0.551

Information Gain = 0.971 - 0.551 = 0.420
```

**Higher Information Gain = Better Split!** 🎯

---

## Gini vs Entropy: Which to Use?

| Aspect | Gini Impurity | Entropy |
|--------|---------------|---------|
| **Range** | 0 to 0.5 | 0 to 1 |
| **Computation** | Faster (no log) | Slower (requires log) |
| **Sensitivity** | Less sensitive to changes | More sensitive |
| **Default in sklearn** | ✅ Yes | Available |
| **Interpretation** | Misclassification probability | Information/disorder |
| **Results** | Usually similar | Usually similar |

### Recommendation
- **Use Gini** for faster computation (default in most libraries)
- **Use Entropy** when you want more balanced trees
- In practice, both give similar results

---

## Advantages of Decision Trees

✅ **Easy to Understand**: Visual and intuitive  
✅ **No Feature Scaling**: Works with raw data  
✅ **Handles Non-linear Relationships**: Can capture complex patterns  
✅ **Handles Both Numerical and Categorical Data**  
✅ **Feature Importance**: Automatically ranks feature importance  
✅ **No Assumptions**: Doesn't assume linear relationships  
✅ **White Box Model**: Easy to explain predictions  

---

## Disadvantages of Decision Trees

❌ **Overfitting**: Can create overly complex trees  
❌ **Instability**: Small data changes can change tree structure  
❌ **Biased**: Favors features with more levels  
❌ **Not Ideal for Continuous Values**: Better for classification  
❌ **Greedy Algorithm**: May not find global optimum  

---

## Overfitting and Solutions

### What is Overfitting?
When the tree becomes too deep and memorizes training data, performing poorly on new data.

### Solutions (Pruning)

#### 1. **Pre-Pruning** (Early Stopping)
Stop tree growth early with parameters:
- `max_depth`: Limit tree depth
- `min_samples_split`: Minimum samples required to split
- `min_samples_leaf`: Minimum samples in leaf nodes
- `max_leaf_nodes`: Maximum number of leaf nodes

#### 2. **Post-Pruning**
Build full tree, then remove branches that don't improve validation accuracy

---

## Use Cases

### Classification Trees
- Customer churn prediction
- Loan approval decisions
- Medical diagnosis
- Email spam detection
- Fraud detection

### Regression Trees
- House price prediction
- Sales forecasting
- Stock market analysis

---

## Decision Tree Example

```
                    [Root: Salary]
                          |
              Yes (>50K) / \ No (≤50K)
                        /   \
                  [Age]       [No Loan]
                   / \
        Yes(>30) /   \ No(≤30)
                /     \
         [Loan]      [No Loan]
```

---

## Key Concepts Summary

1. **Node Types**:
   - **Root Node**: Top node (entire dataset)
   - **Internal Nodes**: Decision nodes (split data)
   - **Leaf Nodes**: Terminal nodes (final predictions)

2. **Splitting Process**:
   - Evaluate all features and split points
   - Choose split with highest Information Gain (or lowest Gini)
   - Repeat recursively

3. **Stopping Criteria**:
   - All samples in node belong to same class
   - Maximum depth reached
   - Minimum samples per node reached
   - No information gain from further splits

---

## Scikit-learn Parameters

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',        # or 'entropy'
    max_depth=None,          # Maximum depth
    min_samples_split=2,     # Min samples to split
    min_samples_leaf=1,      # Min samples in leaf
    max_features=None,       # Max features to consider
    random_state=42
)
```

---

## Visualization

Decision trees can be visualized using:
- `sklearn.tree.plot_tree()`
- `graphviz` library
- `dtreeviz` library

This makes them highly interpretable!

---

## Related Algorithms

- **Random Forest**: Ensemble of many decision trees
- **Gradient Boosting**: Sequential trees correcting errors
- **XGBoost, LightGBM**: Advanced boosting algorithms
- **CART** (Classification and Regression Trees)

---

## Important Notes

⚠️ **Always split data into train/test sets to avoid overfitting**  
⚠️ **Use cross-validation to tune hyperparameters**  
⚠️ **Consider ensemble methods for better performance**  
⚠️ **Decision trees work best with pruning or as part of ensembles**

---

## Quick Comparison

| Purity Level | Gini Value | Entropy Value | Interpretation |
|--------------|-----------|---------------|----------------|
| Pure (all same class) | 0.0 | 0.0 | ✅ Perfect split |
| Mixed (50-50) | 0.5 | 1.0 | ❌ Worst split |
| Mostly pure (90-10) | 0.18 | 0.47 | ✅ Good split |

**Lower values = Better purity = Better split!**