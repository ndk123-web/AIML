## SVM (Support Vector Machine)

SVM ek supervised machine learning algorithm hai jo mostly **classification** ke liye use hota hai (regression ke liye bhi variant hota hai: SVR).

Simple idea: algorithm ek aisi boundary (line/plane) dhoondhta hai jo classes ko separate kare aur unke beech ka gap (margin) maximum ho.

---

## 1. Core Intuition

2D me socho: do classes ke points hain. SVM ek line choose karta hai jo in dono classes ko alag kare.

Lekin sirf separate karna enough nahi hai. SVM chahta hai:

- boundary classes se jitni door ho sake utni door ho
- closest points se distance maximize ho

Isko hi **maximum margin classifier** bolte hain.

---

## 2. Important Terms

- **Hyperplane**:
	- 2D me line
	- 3D me plane
	- higher dimension me decision boundary

- **Margin**:
	- hyperplane aur nearest data points ke beech ka distance
	- SVM margin ko maximize karta hai for better generalization

- **Support Vectors**:
	- woh points jo boundary ke sabse close hote hain
	- inhi points se final hyperplane decide hota hai
	- agar support vectors hata do, boundary change ho sakti hai

---

## 3. Hard Margin vs Soft Margin

### Hard Margin SVM

- assume karta hai data perfectly linearly separable hai
- koi misclassification allow nahi karta
- practical real-world noisy data me rare hai

### Soft Margin SVM

- kuch misclassification allow karta hai
- objective: large margin + low classification error ka balance
- real datasets me mostly yehi use hota hai

Is balance ko control karta hai parameter **C**.

- `C` high -> misclassification ka penalty high -> model strict ho jata hai (overfitting risk)
- `C` low -> penalty low -> margin wider, thodi mistakes allowed (better generalization possible)

---

## 4. Linear vs Non-Linear Separation

Agar data linearly separable nahi hai to simple straight line se split mushkil hota hai.

Yahan aata hai **Kernel Trick**:

- data ko higher-dimensional space me map kiya jata hai
- us space me linear separator mil sakta hai
- mapping explicitly compute nahi karte, kernel function se efficiently similarity calculate karte hain

Common kernels:

- **Linear Kernel**: jab data roughly linear ho
- **RBF (Gaussian) Kernel**: non-linear patterns ke liye most common default
- **Polynomial Kernel**: jab feature interactions polynomial nature ke ho
- **Sigmoid Kernel**: less common in practice

---

## 5. Loss Function (High Level)

SVM training me typical objective hota hai:

- margin maximize karna
- classification error control karna

Iske liye commonly **hinge loss** use hota hai. Kuch implementations squared hinge bhi use kar sakti hain.

Hinge loss intuition:

- correctly and confidently classified point -> low/zero loss
- boundary ke wrong side ya near boundary point -> higher loss

---

## 6. Multi-Class Classification in SVM

Basic SVM binary classifier hai, lekin multi-class ke liye wrappers use hote hain:

- **One-vs-Rest (OvR)**: har class vs baaki sab
- **One-vs-One (OvO)**: har pair of classes ke liye classifier

`scikit-learn` me yeh internally handle ho jata hai.

---

## 7. Why Feature Scaling Is Important

SVM distance-based concept pe kaam karta hai, isliye:

- features ka scale different hua to boundary biased ho sakti hai
- hamesha `StandardScaler` ya similar scaling lagana recommended hai

Especially RBF kernel ke sath scaling almost mandatory hai.

---

## 8. Advantages

- high-dimensional data (many features) par strong performance
- clear margin-based optimization
- kernel trick se non-linear boundaries handle karta hai
- small to medium sized datasets me kaafi effective

---

## 9. Limitations

- very large datasets par training slow ho sakti hai
- parameter tuning (`C`, `gamma`, kernel) sensitive hota hai
- output directly probability nahi hota (extra calibration required)
- noisy/overlapping classes me performance drop ho sakta hai

---

## 10. Important Hyperparameters

- `C`: regularization strength (error vs margin trade-off)
- `kernel`: linear, rbf, poly, sigmoid
- `gamma` (RBF/poly/sigmoid): influence of single training point
	- high gamma -> complex decision boundary (overfitting risk)
	- low gamma -> smoother boundary (underfitting risk)
- `degree` (poly kernel): polynomial degree

---

## 11. Quick scikit-learn Example

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# X = features, y = labels
X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
		("scaler", StandardScaler()),
		("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

## 12. Best Practices

- pehle baseline linear kernel try karo
- data non-linear lage to RBF kernel try karo
- `GridSearchCV` / `RandomizedSearchCV` se `C`, `gamma`, `kernel` tune karo
- imbalanced dataset ho to `class_weight='balanced'` consider karo
- performance evaluate karte time sirf accuracy nahi, `precision`, `recall`, `f1-score` bhi dekho

---

## 13. One-Line Summary

SVM ek margin-maximizing classifier hai jo support vectors ke basis par decision boundary banata hai, aur kernels ki help se non-linear data ko bhi effectively classify kar sakta hai.