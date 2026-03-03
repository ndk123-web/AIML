# Spam Classification using Naive Bayes - Detailed Guide

## 📌 Project Overview

Yeh project Naive Bayes algorithm ka use karke **spam messages ko classify** karne ke liye banaya gaya hai. Dataset mein SMS messages hain jo spam ya legitimate hai.

---

## 🎯 Project ke Main Components

### 1. **Data Loading aur Preprocessing**
```
- spam.csv file ko load karte hain pandas se
- Two columns: 'Message' (text) aur 'Category' (spam/ham)
- Category ko LabelEncoder se numeric format mein convert karte hain (0 = ham, 1 = spam)
```

### 2. **Feature Extraction**
```
- CountVectorizer use karte hain text ko numbers mein convert karne ke liye
- Har unique word ko count karte hain aur bag-of-words representation banate hain
- Example: "Free money now" → [0, 1, 0, 0, 1, 1, 0, ...] (vocabulary ke hisaab se)
```

### 3. **Train-Test Split**
```
- Data ko 67% training aur 33% testing mein divide karte hain
- Models ko training data par sikhate hain aur testing data par evaluate karte hain
```

### 4. **Naive Bayes Variants**
Project mein 3 types of Naive Bayes use hue hain:

| Type | Use Case | Accuracy |
|------|----------|----------|
| **BernoulliNB** | Binary features (0/1) - iska word aata hai ya nahi | 0.38% ❌ |
| **MultinomialNB** | Discrete counts (word frequency) | 7.99% ❌ |
| **GaussianNB** | Continuous features (assumed normal distribution) | - |

### 5. **Pipeline**
```
- CountVectorizer + BernoulliNB dono ko ek saath chain karte hain
- Automatically features convert + classify karte hain
- Code: Pipeline([('vectorizer', CountVectorizer()), ('nb', BernoulliNB())])
```

---

## ⚠️ 🔴 CRITICAL ISSUE - WHY ACCURACY SO LOW?

### **The Main Problem:**

```python
# ❌ WRONG CODE (Line 86-87)
X = df.drop(['output'],axis=1)
y = df["Message"]  

# ✅ CORRECT CODE SHOULD BE
X = df[['Message']]  # Features (text messages)
y = df['output']      # Labels (0 for ham, 1 for spam)
```

**Kya galat tha?**
1. `y = df["Message"]` - ye **text messages** ko label maan rahe the (bilkul hi galat!)
2. Model ko bataya: "Yeh message predict karo... message se" 
3. Ye nonsensical task hai! Model kabhi predict nahi kar sakta

**Iska mtlb:**
- Model ne **spam/ham labels** nahi dekhe training time par
- Instead, usne message text dekhi jo same test data mein bhi hogi
- Random predictions de rahe hain, isliye accuracy 0.38% hai (random guessing se bhi worse!)

---

## 📊 Expected vs Actual Accuracy

### **Actual Results (Current - WRONG)**
```
BernoulliNB:  0.38% ❌
MultinomialNB: 7.99% ❌
Pipeline:      0.38% ❌
```

### **Expected Results (After Fix)**
```
BernoulliNB:  ~97-98% ✅
MultinomialNB: ~98-99% ✅ (Best for text data)
```

---

## 🛠️ Fix kaise karein?

### **Step 1: Correct the X aur y assignment**
```python
# Current (Line 86-87) - DELETE these lines
# X = df.drop(['output'],axis=1)
# y = df["Message"]

# Add these correct lines:
X = df[['Message']]  # Features = message text
y = df['output']     # Labels = spam/ham classification
```

### **Step 2: Update train-test split**
```python
# Current code already correct hai
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

### **Step 3: Pass correct data to models**
```python
# For BernoulliNB
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.Message)  # ✅ ye theek hai
bernouli.fit(X_train_count, y_train)              # ✅ y_train (labels) theek hai

# For Pipeline
clf.fit(X_train.Message, y_train)  # ✅ y_train (labels) theek hai
```

---

## 🧠 Naive Bayes Algorithm Explained (Hindi)

### **Bayes Theorem**
```
P(Spam | Message) = P(Message | Spam) × P(Spam) / P(Message)
```

**In Hinglish:**
- **P(Spam | Message)** = Message dekh kar spam hone ki probability
- **P(Message | Spam)** = Agar spam hai to ye message aane ki probability
- **P(Spam)** = Overall khitra ka spam hai
- **P(Message)** = Ye specific message aane ki probability

### **Kaise kaam karta hai?**
1. **Training:** Har word ko count karte hain spam aur ham messages mein
2. **Testing:** Naa message ke liye haar word dekh kar probability calculate karte hain
3. **Decision:** Agar spam probability zyada hai → Spam, else → Ham

---

## 📁 Data Structure

```
spam.csv
├── Column 1: Message (text)
│   ├── "Go until jurong point, crazy... Available only..." (Ham)
│   ├── "Free entry in 2 a wkly comp..." (Spam ✓)
│   └── "U dun say so early hor..." (Ham)
│
└── Column 2: Category (label)
    ├── "ham" → 0
    └── "spam" → 1
```

---

## 🎓 Key Learnings

### **Do's ✅**
1. Always ensure `X` (features) aur `y` (labels) correctly assigned hain
2. Text preprocessing important hai (lemmatization, stop words removal)
3. MultinomialNB text classification ke liye best hai
4. Evaluate karte time confusion matrix dekho (FP vs FN)

### **Don'ts ❌**
1. Features aur labels ko confuse na karo
2. Raw text ko directly model ko na do (vectorize karo pehle)
3. Small dataset par evaluation trust mat karo (cross-validation use karo)
4. Accuracy hi na dekho - precision, recall, F1-score bhi dekho

---

## 📈 Improvements for Better Accuracy

```python
# 1. Text Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# 2. Better model parameters
clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('nb', MultinomialNB(alpha=0.1))
])

# 3. Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)

# 4. Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 🔗 Types of Naive Bayes

| Naive Bayes Type | Feature Type | Use Case |
|---|---|---|
| **Gaussian** | Continuous values | Age, salary, etc. |
| **Multinomial** | Discrete counts | Text, document classification (BEST FOR SPAM) |
| **Bernoulli** | Binary values | Yes/No, word present/absent |

**Spam detection ke liye: MultinomialNB best hai!** 📧

---

## ✅ Summary

| Aspect | Problem | Solution |
|--------|---------|----------|
| **Accuracy** | 0.38% - 7.99% (Terrible ❌) | Fix X aur y assignment |
| **Root Cause** | y = Message (text) instead of y = output (labels) | Use y = df['output'] |
| **Expected Result** | - | 97-99% accuracy after fix |
| **Best Model** | BernoulliNB not suitable | Use MultinomialNB for text |

---

## 📚 References

- Naive Bayes Theory: P(A\|B) = P(B\|A) × P(A) / P(B)
- CountVectorizer: Converts text to word counts
- Pipeline: Chains preprocessing + classification steps

