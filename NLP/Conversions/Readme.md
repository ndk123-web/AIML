# Text Vectorization & Word Embeddings (TF-IDF vs. Word2Vec)

A comprehensive guide and technical reference covering the foundational concepts, mathematical formulations, implementation details, and differences between classical statistical vectorization (**TF-IDF**) and dense distributed semantic embeddings (**Word2Vec**) based on the notebooks in this directory.

---

## Table of Contents
1. [The Text Representation Problem](#1-the-text-representation-problem)
2. [Evolution of Text Representations](#2-evolution-of-text-representations)
3. [TF-IDF (Term Frequency - Inverse Document Frequency)](#3-tf-idf-term-frequency---inverse-document-frequency)
   * [3.1 Core Mathematical Formula](#31-core-mathematical-formula)
   * [3.2 Step-by-Step Corpus Walkthrough](#32-step-by-step-corpus-walkthrough)
   * [3.3 What is a Sparse Matrix? (`scipy.sparse.csr_matrix` vs. `.toarray()`)](#33-what-is-a-sparse-matrix-scipysparsecsr_matrix-vs-toarray)
   * [3.4 Does TF-IDF Require Training?](#34-does-tf-idf-require-training)
   * [3.5 Scikit-Learn Mechanics (`fit_transform` & Data Leakage Prevention)](#35-scikit-learn-mechanics-fit_transform--data-leakage-prevention)
   * [3.6 Strengths & Limitations](#36-strengths--limitations)
4. [Word2Vec (Dense Semantic Embeddings)](#4-word2vec-dense-semantic-embeddings)
   * [4.1 Core Intuition: The Distributional Hypothesis](#41-core-intuition-the-distributional-hypothesis)
   * [4.2 Self-Supervision: The Automatic Label Generator](#42-self-supervision-the-automatic-label-generator)
   * [4.3 Under the Hood: How a Word's Vector is Learned Step-by-Step](#43-under-the-hood-how-a-words-vector-is-learned-step-by-step)
   * [4.4 Architectures: CBOW vs. Skip-Gram](#44-architectures-cbow-vs-skip-gram)
   * [4.5 Gensim Hyperparameter Deep Dive](#45-gensim-hyperparameter-deep-dive)
   * [4.6 Why Use Cosine Similarity If Embeddings Are Already Static?](#46-why-use-cosine-similarity-if-embeddings-are-already-static)
   * [4.7 Semantic Vector Arithmetic (Analogies & Similarity)](#47-semantic-vector-arithmetic-analogies--similarity)
5. [End-to-End Pipeline Architecture](#5-end-to-end-pipeline-architecture)
6. [Comprehensive Comparison Tables](#6-comprehensive-comparison-tables)
7. [Decision Matrix: When to Use What?](#7-decision-matrix-when-to-use-what)
8. [Notebook Index](#8-notebook-index)

---

## 1. The Text Representation Problem

Machine Learning models and Neural Networks cannot directly process raw text strings like `"machine learning"`. They operate solely on numerical tensors ($\mathbb{R}^d$). 

The objective of **Text Vectorization** is to map discrete symbols (characters, words, sentences, or documents) into numerical vectors in a continuous geometric space while preserving as much linguistic and semantic signal as possible.

```
                    ┌──────────────────────────────┐
                    │ Raw String: "Machine Learn"  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼ [Vectorization / Embedding]
                    ┌──────────────────────────────┐
                    │ Numerical Vector / Tensor    │
                    │ e.g. [0.428, 0.0, 0.562, ...]│
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   ML / Deep Learning Model   │
                    └──────────────────────────────┘
```

---

## 2. Evolution of Text Representations

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 1. One-Hot Encoding (OHE)                                                        │
│    - Sparse, orthogonal vectors of size V (Vocabulary). 0 semantic information.  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 2. Bag of Words (BoW) & CountVectorizer                                          │
│    - Counts word occurrences per document. Ignores word order and significance.  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 3. TF-IDF (Statistical Term Weighting)                                           │
│    - Weighs words by document frequency vs. corpus rarity. Still sparse & 0-dim. │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 4. Word2Vec & Static Distributed Embeddings (2013)                               │
│    - Dense, low-dimensional vectors (d=100-300) capturing semantic similarity.  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 5. Contextualized Embeddings (BERT / Transformers) (2018+)                       │
│    - Dynamic vectors generated on-the-fly depending on full sentence context.   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. TF-IDF (Term Frequency - Inverse Document Frequency)

**TF-IDF** is a classical statistical weighting technique used to quantify how important a word is to a specific document within a collection (corpus).

### 3.1 Core Mathematical Formula

The TF-IDF score of a term $t$ in a document $d$ within a corpus $D$ is computed as:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

#### A. Term Frequency ($\text{TF}$)
Measures the relative frequency of term $t$ in document $d$:
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
*(In standard raw frequency, it is simply the count of term $t$ in document $d$.)*

#### B. Inverse Document Frequency ($\text{IDF}$)
Measures how rare or informative term $t$ is across the entire corpus of $N$ documents:
* Words appearing in almost every document (e.g., `"the"`, `"is"`, `"and"`) get an $\text{IDF} \to 0$.
* Words appearing in only a few documents get a high $\text{IDF}$.

**Scikit-Learn Smooth IDF Formula:**
$$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
* $N$: Total number of documents in the corpus.
* $\text{DF}(t)$: Number of documents containing term $t$.
* The `+ 1` terms prevent division by zero (smoothing) and ensure that terms present in all documents do not get completely zeroed out.

#### C. L2 Euclidean Normalization
Scikit-Learn applies row-wise Euclidean $L_2$ normalization so that document length differences do not distort vector magnitudes:
$$v_{\text{normalized}} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^{|V|} v_i^2}}$$

---

### 3.2 Step-by-Step Corpus Walkthrough

From [1_tf_idf_vectorization.ipynb](file:///d:/ML/ML-Learning/NLP/Conversions/1_tf_idf_vectorization.ipynb):

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Machine learning is fascinating.",
    "Deep learning and machine learning are related.",
    "Natural language processing uses text."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
```

#### 1. Vocabulary Extraction (`get_feature_names_out()`):
Scikit-Learn lowercases, tokenizes, strips punctuation, and sorts unique words alphabetically:
```python
['and', 'are', 'deep', 'fascinating', 'is', 'language', 'learning', 'machine', 'natural', 'processing', 'related', 'text', 'uses']
```
Total Vocabulary Size ($|V|$) = **13 unique features**.

#### 2. Vector Structure for Sentence 1:
*"Machine learning is fascinating."*

| Feature Index | Word | In Sentence 1? | TF Count | Document Frequency ($\text{DF}$) | TF-IDF (L2 Normalized) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | `and` | No | 0 | 1 | `0.000000` |
| `1` | `are` | No | 0 | 1 | `0.000000` |
| `2` | `deep` | No | 0 | 1 | `0.000000` |
| `3` | `fascinating` | **Yes** | 1 | 1 (Rare) | **`0.562829`** |
| `4` | `is` | **Yes** | 1 | 1 (Rare) | **`0.562829`** |
| `5` | `language` | No | 0 | 1 | `0.000000` |
| `6` | `learning` | **Yes** | 1 | 2 (Common) | **`0.428046`** |
| `7` | `machine` | **Yes** | 1 | 2 (Common) | **`0.428046`** |
| `8` | `natural` | No | 0 | 1 | `0.000000` |
| `9` | `processing` | No | 0 | 1 | `0.000000` |
| `10` | `related` | No | 0 | 1 | `0.000000` |
| `11` | `text` | No | 0 | 1 | `0.000000` |
| `12` | `uses` | No | 0 | 1 | `0.000000` |

```python
# Output Array for Sentence 1:
[0.0, 0.0, 0.0, 0.5628291, 0.5628291, 0.0, 0.42804604, 0.42804604, 0.0, 0.0, 0.0, 0.0, 0.0]
```

> **Key Takeaway:** Notice that `'fascinating'` ($0.5628$) has a **higher** TF-IDF weight than `'learning'` ($0.4280$) because `'learning'` appears in 2 documents (lower IDF), whereas `'fascinating'` appears in only 1 document (higher IDF / more informative).

---

### 3.3 What is a Sparse Matrix? (`scipy.sparse.csr_matrix` vs. `.toarray()`)

A **Sparse Matrix** is a matrix in which the overwhelming majority of entries are **zero**.

In NLP, if your vocabulary contains **50,000 words** and an average sentence contains **5 words**, a standard dense row representation would store **50,000 floating-point numbers** where **49,995 entries are `0.0`**.

```
Dense Matrix (Wasteful):
Sentence 1: [ 0.0, 0.0, ..., 0.81 (word #42), ..., 0.58 (word #319), ..., 0.0 ] (50,000 floats!)

Compressed Sparse Row (CSR) (Efficient):
Sentence 1: Coordinates only -> { Index 42: 0.81, Index 319: 0.58 }
```

* **RAM Optimization:** Storing millions of `0.0`s as full 64-bit NumPy float arrays would instantly exhaust system RAM ($\approx 50,000 \times 100,000 \times 8 \text{ bytes} \approx 40 \text{ GB}$).
* **`scipy.sparse.csr_matrix`:** Scikit-Learn stores **only non-zero values and their column coordinate indices**.
* **`.toarray()` / `.todense()`:** Unpacks the compressed coordinate structure into a full 2D NumPy array for display, debugging, or plotting.

---

### 3.4 Does TF-IDF Require Training?

**Yes, but it is pure deterministic statistical counting, NOT neural network gradient descent.**

* `fit()` scans all documents in the corpus to count document frequencies ($\text{DF}(w)$) and compute the static mathematical formula:
  $$\text{IDF}(w) = \ln\left(\frac{1 + N}{1 + \text{DF}(w)}\right) + 1$$
* **No Learnable Parameters:** There are no stochastic weights, no learning rate, no epochs, and no loss functions. It is a deterministic, exact closed-form calculation.

---

### 3.5 Scikit-Learn Mechanics (`fit_transform` & Data Leakage Prevention)

* **`fit(corpus)`**: Scans the training corpus, builds the vocabulary dictionary mapping each word to an index, and calculates the global $\text{IDF}$ vector for all words.
* **`transform(corpus)`**: Converts incoming raw text documents into numerical vectors using the *already learned* vocabulary and $\text{IDF}$ weights.
* **`fit_transform(corpus)`**: Executes both operations in a single optimized pass on training data.

> **Production Rule:** Always call `fit_transform(X_train)` on your training data, but **only** `transform(X_test)` on test/validation data. Fitting on test data causes **Data Leakage**, artificially boosting test scores by leaking future document frequency statistics.

---

### 3.6 Strengths & Limitations

#### Strengths:
* **Fast and Lightweight:** Linear computation time $\mathcal{O}(N \times L)$.
* **Interpretable:** Every dimension corresponds to a specific known word in the vocabulary.
* **Effective for Search / Keyword Matching:** Penalizes meaningless stopwords naturally.

#### Limitations:
* **High Dimensionality & Sparsity:** Vector size grows with every new unique word ($|V|$).
* **Zero Semantic Awareness:** Treats `"automobile"` and `"car"` as completely independent, orthogonal dimensions with $0.0$ cosine similarity.
* **Ignores Word Ordering:** `"dog bites man"` and `"man bites dog"` produce the exact same TF-IDF vector (Bag-of-Words assumption).

---

## 4. Word2Vec (Dense Semantic Embeddings)

Developed by *Tomas Mikolov et al. (Google, 2013)*, **Word2Vec** revolutionized NLP by learning dense, distributed, low-dimensional continuous vector representations ($d \in [50, 300]$) for words from unstructured text.

---

### 4.1 Core Intuition: The Distributional Hypothesis

Word2Vec is built upon the famous linguistic principle from *J.R. Firth (1957)*:
> *"You shall know a word by the company it keeps."*

Words that appear in similar surrounding contexts share similar semantic meanings.

---

### 4.2 Self-Supervision: The Automatic Label Generator

* **Self-Supervised Learning:** You do not need human-annotated labels. The raw text corpus automatically creates its own pseudo-supervision targets by sliding a context window over the text:
  * **CBOW:** Context words $\to$ Target word.
  * **Skip-Gram:** Target word $\to$ Surrounding context words.

---

### 4.3 Under the Hood: How a Word's Vector is Learned Step-by-Step

A common point of confusion is: *"Are a word's vector and the neural network's weights two different things?"*

> **The Fundamental Truth:** A word's 10-dimensional vector and the neural network's connection weights are the **EXACT SAME THING!**

#### 1. Network Architecture & Connection Diagram
Assume vocabulary size $V = 500$ unique words and embedding dimension `vector_size = 10` ($10$ Hidden Neurons).

```
Input Layer (One-Hot: Size 500)                Hidden Layer (10 Neurons)
┌──────────────────────────────┐              ┌────────────────────────┐
│ Node 0 ("king" = 1)          │══════════════╡ Neuron 0               │
│                              │  10 Weights  │ Neuron 1               │
│ Node 1 ("queen" = 0)         │  (w0,0..w0,9)│ Neuron 2               │
│ Node 2 ("man" = 0)           │              │ ...                    │
│ ...                          │              │ Neuron 9               │
│ Node 499 ("apple" = 0)       │              └────────────────────────┘
└──────────────────────────────┘
```

The connection between the 500 Input Nodes and the 10 Hidden Neurons forms the **Weight Matrix ($W_{\text{embed}}$) of shape $(500 \times 10)$**.
* **Row 0** of $W_{\text{embed}}$: The 10 connection weights for `"king"`.
* **Row 1** of $W_{\text{embed}}$: The 10 connection weights for `"queen"`.
* **Row 499** of $W_{\text{embed}}$: The 10 connection weights for `"apple"`.

#### 2. Forward Pass: Exact Row Selection
When `"king"` enters the network (`Node 0 = 1`, all other nodes $= 0$):
$$[1, 0, 0, \dots, 0]_{(1 \times 500)} \times W_{\text{embed}(500 \times 10)} = \text{Row 0 of } W_{\text{embed}} \ (1 \times 10)$$
The values reaching the 10 hidden neurons are literally **Row 0 of $W_{\text{embed}}$**!

#### 3. Prediction & Backpropagation Cycle (1-by-1 Training):

```
                        ┌──────────────────────────────────────────────────────────┐
                        │             WORD-BY-WORD TRAINING CYCLE                  │
                        └──────────────────────────────────────────────────────────┘

 Step 1: Input "king"   ──► Forward Pass ──► Output Predicts "rules" ──► Loss Computed
                                                                               │
                                                                               ▼ (Backpropagation)
                            Gradient Descent updates ONLY Row 0 (king's 10 weights!)

 Step 2: Input "queen"  ──► Forward Pass ──► Output Predicts "rules" ──► Loss Computed
                                                                               │
                                                                               ▼ (Backpropagation)
                            Gradient Descent updates ONLY Row 1 (queen's 10 weights!)

 Step 3: Input "apple"  ──► Forward Pass ──► Output Predicts "fruit" ──► Loss Computed
                                                                               │
                                                                               ▼ (Backpropagation)
                            Gradient Descent updates ONLY Row 499 (apple's 10 weights!)
```

#### 4. Post-Training: Extracting Static Embeddings
After all epochs complete:
1. The output classification layer is discarded.
2. The **Weight Matrix $W_{\text{embed}}$ ($500 \times 10$)** is frozen as the **Static Embedding Table** (`model.wv`).
3. Calling `model.wv['king']` performs a direct table lookup: it reads **Row 0** and returns those **10 learned floats**!

Because `"king"` and `"queen"` regularly predicted identical context words (*"rules"*, *"crown"*, *"palace"*), backpropagation pushed their 10 weights into nearly identical coordinate patterns. Unrelated words (*"apple"*) were pushed to completely different coordinates.

---

### 4.4 Architectures: CBOW vs. Skip-Gram

Word2Vec provides two distinct training architectures:

```
        CBOW (Continuous Bag of Words)                 Skip-Gram Architecture
         (Predict Target from Context)              (Predict Context from Target)

          Context Words                                      Context Words
         ┌─────────────┐                                    ┌─────────────┐
         │  w(t-2)     │──┐                              ┌─▶│  w(t-2)     │
         └─────────────┘  │                              │  └─────────────┘
         ┌─────────────┐  │   ┌───────────┐   ┌────────┐ │  ┌─────────────┐
         │  w(t-1)     │──┼──▶│ Projection│──▶│  w(t)  │ │  │  w(t-1)     │
         └─────────────┘  │   │ (Average) │   │ (Target)│ │  └─────────────┘
         ┌─────────────┐  │   └───────────┘   └────────┘ │  ┌─────────────┐
         │  w(t+1)     │──┤                              ├──│    w(t)     │ (Input Target)
         └─────────────┘  │                              │  └─────────────┘
         ┌─────────────┐  │                              │  ┌─────────────┐
         │  w(t+2)     │──┘                              ├─▶│  w(t+1)     │
         └─────────────┘                                 │  └─────────────┘
                                                         │  ┌─────────────┐
                                                         └─▶│  w(t+2)     │
                                                            └─────────────┘
```

#### A. CBOW (Continuous Bag of Words) — `sg=0`
* **Goal:** Given surrounding context words within window $C$ ($\{w_{t-c}, \dots, w_{t+c}\}$), predict the central target word $w_t$.
* **Mechanism:** Averages the input embedding vectors of all context words and passes them through a linear layer to predict $w_t$.
* **Characteristics:** Fast training; smooths over statistical noise; excels on frequent words and larger corpora.

#### B. Skip-Gram — `sg=1`
* **Goal:** Given a single center target word $w_t$, predict each surrounding context word within window $c$.
* **Mechanism:** Uses the target word's embedding vector to compute probabilities across candidate context words using softmax or negative sampling.
* **Characteristics:** Slower training; handles small datasets exceptionally well; captures fine-grained semantics and rare words significantly better.

---

### 4.5 Gensim Hyperparameter Deep Dive

From [2_word_2_vec.ipynb](file:///d:/ML/ML-Learning/NLP/Conversions/2_word_2_vec.ipynb):

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "empire"],
    ["a", "man", "walks", "into", "a", "bar"],
    ["a", "woman", "walks", "into", "a", "restaurant"]
]

model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1)
```

| Hyperparameter | Value in Notebook | Standard Production Value | Description |
| :--- | :--- | :--- | :--- |
| **`vector_size`** | `10` | `100` to `300` | Dimensionality of the resulting dense embedding vector per word (Hidden Layer Neuron count). |
| **`window`** | `2` | `5` (CBOW) / `10` (Skip-Gram) | Maximum distance (left and right) between the current target word and predicted context words. |
| **`min_count`** | `1` | `5` | Words appearing fewer than `min_count` times are discarded from the vocabulary to reduce noise. |
| **`sg`** | `1` | `1` (Skip-Gram) or `0` (CBOW) | Training algorithm selector: `1` for Skip-Gram, `0` for CBOW. |
| **`workers`** | Default | `4` to `8` | Number of CPU worker threads used for parallel training. |
| **`epochs`** | Default (`5`) | `10` to `30` | Number of training iterations over the corpus. |

---

### 4.6 Why Use Cosine Similarity If Embeddings Are Already Static?

You do **NOT** need Cosine Similarity to generate or fetch embeddings. The embedding table already exists as frozen static vectors in memory.

#### 1. Embedding Retrieval (Direct Index Lookup $\mathcal{O}(1)$):
Given the token `"king"`, we retrieve its vector directly from the weight matrix:
```python
vector = model.wv['king'] # Shape: [10] or [300]
```
No cosine calculation or geometric distance is computed here.

#### 2. Why Cosine Similarity IS Used (Post-Training Search & Reasoning):
* **Semantic Nearest Neighbors:** To answer queries like *"Which word in the dictionary is closest in meaning to 'king'?"*, the system computes the cosine similarity between `vector('king')` and all other rows in the vocabulary table:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \cos(\theta)$$
  ```python
  print(model.wv.most_similar('king', topn=2))
  # Output: [('empire', 0.5111), ('woman', 0.3861)]
  ```
* **Analogy Verification & Vector Algebra:** To evaluate conceptual relationships:
  $$\vec{v} = \vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}}$$
  The system computes cosine similarity between resulting vector $\vec{v}$ and every word in the dictionary to find that $\vec{\text{queen}}$ has the highest similarity score.

```python
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("Arithmetic Result:", result)
```

---

### 4.7 Semantic Vector Arithmetic (Analogies & Similarity)

Because Word2Vec maps semantic properties to linear vector directions:
* **Gender Direction:** $\vec{\text{King}} - \vec{\text{Queen}} \approx \vec{\text{Man}} - \vec{\text{Woman}}$
* **Capital City Direction:** $\vec{\text{Paris}} - \vec{\text{France}} \approx \vec{\text{Rome}} - \vec{\text{Italy}}$

---

## 5. End-to-End Pipeline Architecture

```text
1. Raw Text / Tokens ──► [Static Embedding Matrix Lookup: V × d] ──► Dense Vector (e.g., 100 floats)
                                                                            │
                                      ┌─────────────────────────────────────┴─────────────────────────────────────┐
                                      ▼                                                                           ▼
           [Pass dense vectors to PyTorch RNN / LSTM / Transformer]            [Compute Cosine Similarity for Semantic Search & Analogies]
           (e.g., Sentiment Analysis, Named Entity Recognition)                (e.g., Vector DBs, most_similar, Recommendation Engines)
```

---

## 6. Comprehensive Comparison Tables

### Comparison 1: TF-IDF vs. Word2Vec

| Attribute | TF-IDF (`TfidfVectorizer`) | Word2Vec (`gensim.models.Word2Vec`) |
| :--- | :--- | :--- |
| **Representation Type** | **Sparse** (Mostly 0s, size = $|V|$) | **Dense** (Fixed small floats, $d \approx 100\text{--}300$) |
| **Granularity** | Document-level or Sentence-level vector | Word-level embedding matrix |
| **Underlying Mechanism** | Statistical counting + smooth logarithmic inverse frequency | 2-Layer Neural Network (CBOW / Skip-Gram) |
| **Training Type** | Deterministic closed-form formula (No loss/epochs) | Self-Supervised Neural Network (Gradient Descent) |
| **Semantic Relationships**| ❌ **None** (Orthogonal dimensions) | ✅ **Rich** (Synonyms & analogies cluster together) |
| **Word Order & Context** | ❌ **Ignored** (Bag of Words) | ✅ **Preserved** (Learned from local window sliding) |
| **Dimensionality** | High & Scalable ($10,000\text{--}100,000+$) | Fixed & Compact ($50\text{--}300$) |
| **Storage Format** | `scipy.sparse.csr_matrix` (Non-zero coordinates) | Dense 2D Float Tensor ($V \times d$) |
| **Out-Of-Vocabulary (OOV)**| Drops unseen words during `transform()` | Cannot generate vectors for unseen words |

---

### Comparison 2: CBOW vs. Skip-Gram

| Feature | CBOW (`sg=0`) | Skip-Gram (`sg=1`) |
| :--- | :--- | :--- |
| **Prediction Direction** | Context Words $\to$ Target Word | Target Word $\to$ Context Words |
| **Optimization Target** | Maximize $P(w_t \mid w_{t-c}, \dots, w_{t+c})$ | Maximize $\sum P(w_{t+j} \mid w_t)$ |
| **Training Speed** | Fast ($\approx 3\times$ faster than Skip-Gram) | Slower |
| **Performance on Rare Words** | Average (context averaging dilutes rare signals) | **Superior** (each context pair updated individually) |
| **Optimal Dataset Size** | Large text corpora | Small to medium corpora |
| **Representation Focus** | General syntactic & topical clusters | Fine-grained semantic & relational analogies |

---

## 7. Decision Matrix: When to Use What?

```
                                    ┌─────────────────────────────┐
                                    │    What is your NLP Task?   │
                                    └──────────────┬──────────────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
    ┌─────────────────────────────┐                                 ┌─────────────────────────────┐
    │  Document Classification,   │                                 │  Semantic Search, Analogy,  │
    │  Spam Filtering, Keyword    │                                 │  Clustering, Word Similarity│
    │  Matching, or Small Text    │                                 │  or Feature Representation  │
    └──────────────┬──────────────┘                                 └──────────────┬──────────────┘
                   │                                                               │
                   ▼                                                               ▼
    ┌─────────────────────────────┐                                 ┌─────────────────────────────┐
    │          TF-IDF             │                                 │          Word2Vec           │
    │  - Fast, Interpretable      │                                 │  - Dense, Semantic Vectors  │
    │  - Strong Baseline for ML   │                                 │  - Encodes Word Meaning     │
    └─────────────────────────────┘                                 └─────────────────────────────┘
```

* **Use TF-IDF when:**
  * You need a fast, strong baseline for document classification (e.g., Logistic Regression / Naive Bayes on spam detection).
  * Feature interpretability is critical (you need to know exactly which keywords triggered a prediction).
  * Your corpus is small, and training neural word vectors would overfit or yield poor embeddings.

* **Use Word2Vec when:**
  * You need semantic understanding where synonyms (`"laptop"` and `"notebook"`) should have close vector representations.
  * You are feeding word vectors into sequence models (LSTMs, GRUs, 1D-CNNs) for downstream NLP tasks.
  * You need semantic arithmetic, clustering, or candidate retrieval based on conceptual similarity.

---

## 8. Notebook Index
* [1_tf_idf_vectorization.ipynb](file:///d:/ML/ML-Learning/NLP/Conversions/1_tf_idf_vectorization.ipynb): Scikit-Learn `TfidfVectorizer` pipeline, sparse-to-dense conversion, and feature name mapping.
* [2_word_2_vec.ipynb](file:///d:/ML/ML-Learning/NLP/Conversions/2_word_2_vec.ipynb): Gensim `Word2Vec` Skip-Gram model training, vector inspection, semantic similarity querying, and vector arithmetic.