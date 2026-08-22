<<<<<<< HEAD
## Self Supervised Word Embedding

## Types

1.CBOW (Continuous Bag of Words)

- (Context -> Word)

2. Skip-Gram

- (Word -> Context)
=======
# Self-Supervised Word Embeddings

Self-supervised word embeddings are pretrained representation learning techniques that discover rich semantic and syntactic relationships from massive, unannotated text corpora without requiring manual human labels.

The foundational principle behind self-supervised word embeddings is the **Distributional Hypothesis** (*J.R. Firth, 1957*):
> *"You shall know a word by the company it keeps."*

---

## 🏗️ Major Self-Supervised Embedding Architectures

```
Self-Supervised Static Embeddings
│
├── 1. Word2Vec (Mikolov et al., 2013) — Local Window Prediction
│   ├── CBOW (Continuous Bag of Words): Context Words ➔ Target Word
│   └── Skip-Gram: Target Word ➔ Context Words
│
├── 2. GloVe (Pennington et al., 2014) — Global Matrix Factorization
│   └── Combines global word co-occurrence statistics with local context windows
│
└── 3. FastText (Bojanowski et al., 2017) — Subword Character n-grams
    └── Breaks words into subword chunks to handle morphology & Out-Of-Vocabulary (OOV)
```

---

## 1. Word2Vec: CBOW vs. Skip-Gram

Word2Vec creates dense vector representations such that words appearing in similar contexts end up close to each other in vector space (e.g., $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$).

```
        CBOW Architecture                       Skip-Gram Architecture
(Predict Target from Context)               (Predict Context from Target)

 Context Words                                       Context Words
┌─────────────┐                                     ┌─────────────┐
│  w(t-2)     │──┐                               ┌─▶│  w(t-2)     │
└─────────────┘  │                               │  └─────────────┘
┌─────────────┐  │   ┌───────────┐   ┌─────────┐ │  ┌─────────────┐
│  w(t-1)     │──┼──▶│ Projection│──▶│  w(t)   │ │  │  w(t-1)     │
└─────────────┘  │   │ (Average) │   │(Target) │ │  └─────────────┘
┌─────────────┐  │   └───────────┘   └─────────┘ │  ┌─────────────┐
│  w(t+1)     │──┤                               ├──│  w(t)       │ (Input Target)
└─────────────┘  │                               │  └─────────────┘
┌─────────────┐  │                               │  ┌─────────────┐
│  w(t+2)     │──┘                               ├─▶│  w(t+1)     │
└─────────────┘                                  │  └─────────────┘
                                                 │  ┌─────────────┐
                                                 └─▶│  w(t+2)     │
                                                    └─────────────┘
```

---

### A. CBOW (Continuous Bag of Words)

- **Objective:** Predict the current target word $w_t$ given surrounding context words within a window $C$: $\{w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+c}\}$.
- **Mechanism:**
  1. Look up embedding vectors for all context words.
  2. Average (or sum) the context vectors in the projection layer:
     $$\bar{v} = \frac{1}{2c} \sum_{-c \le j \le c, j \neq 0} v_{w_{t+j}}$$
  3. Multiply by output matrix and pass through softmax to predict probability of the target word $w_t$.
- **Best Suited For:** Faster training; works well with frequent words and larger datasets where averaging smooths out noise.

---

### B. Skip-Gram

- **Objective:** Given a single center target word $w_t$, predict each context word $w_{t+j}$ within window $c$.
- **Mechanism:**
  1. Look up embedding vector $v_{w_t}$ for the center target word.
  2. For each surrounding position in the window, compute score with candidate context words:
     $$P(w_{t+j} \mid w_t) = \frac{\exp({u_{w_{t+j}}^\top v_{w_t}})}{\sum_{w=1}^V \exp({u_w^\top v_{w_t}})}$$
- **Best Suited For:** Smaller datasets; captures rare words and nuanced relationships significantly better than CBOW.

---

## ⚡ Training Optimizations: Negative Sampling & Hierarchical Softmax

Computing standard softmax requires normalizing across the entire vocabulary size $V$ ($\mathcal{O}(V)$), which is prohibitively slow when $V = 100,000+$.

### 1. Negative Sampling (SGNS - Skip-Gram with Negative Sampling)
Instead of multi-class classification across all $V$ words, convert the problem into **binary logistic regression**:
- **Positive Pair:** True (target, context) pair $\to$ Label = 1.
- **Negative Pairs:** Sample $k$ random words from the vocabulary (noise distribution) that do not appear in context $\to$ Label = 0.
- **Complexity:** Reduces computation from $\mathcal{O}(V)$ to $\mathcal{O}(k)$ where $k \approx 5\text{--}20$.

### 2. Hierarchical Softmax
Replaces the flat vocabulary layer with a **Huffman Tree** where words are leaves:
- Frequent words have shorter tree paths.
- Computing probabilities takes $\mathcal{O}(\log_2 V)$ instead of $\mathcal{O}(V)$.

---

## 2. GloVe (Global Vectors for Word Representation)

While Word2Vec relies solely on local context windows, **GloVe** combines the advantages of:
1. **Global matrix factorization** (like LSA/SVD) across the entire corpus.
2. **Local context window methods** (like Word2Vec).

### Key Insight: Ratio of Co-occurrence Probabilities
Let $P_{ik} = P(k \mid i) = \frac{X_{ik}}{X_i}$ be the probability that word $k$ appears in the context of word $i$.

| Ratio: $\frac{P(k \mid \text{ice})}{P(k \mid \text{steam})}$ | $k = \text{solid}$ | $k = \text{gas}$ | $k = \text{water}$ | $k = \text{fashion}$ |
| :--- | :--- | :--- | :--- | :--- |
| **Value** | **Large** ($\gg 1$) | **Small** ($\ll 1$) | $\approx 1$ | $\approx 1$ |

GloVe minimizes the weighted least squares loss:
$$J = \sum_{i,j=1}^V f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
Where $f(X_{ij})$ is a weighting function that prevents very frequent words (like "the", "is") from dominating the loss.

---

## 3. FastText (Subword Embeddings)

Traditional Word2Vec and GloVe assign a single atomic vector to each word, which causes two major problems:
1. **Cannot handle Out-Of-Vocabulary (OOV) words:** Unseen words get a generic `<UNK>` vector.
2. **Ignores morphological structure:** "play", "playing", and "playful" are treated as completely independent vectors.

**FastText** solves this by representing each word as a bag of **character n-grams** (plus the full word):

Example for word `<where>` with $n=3$:
- Character 3-grams: `<wh`, `whe`, `her`, `ere`, `re>`
- Special sequence: `<where>`

The final embedding of a word is the **sum** of all its constituent character n-gram vectors:
$$v_{\text{where}} = \sum_{g \in \mathcal{G}_{\text{where}}} z_g$$

---

## 📊 Comprehensive Comparison Matrix

| Feature | Word2Vec (CBOW) | Word2Vec (Skip-Gram) | GloVe | FastText |
| :--- | :--- | :--- | :--- | :--- |
| **Learning Paradigm** | Local context window | Local context window | Global co-occurrence matrix | Subword character n-grams |
| **Input $\to$ Output** | Context $\to$ Target | Target $\to$ Context | Global co-occurrence counts | Subword n-grams $\to$ Context |
| **Training Speed** | Fast | Moderate | Fast | Slower (more n-grams) |
| **Rare Words** | Average | Good | Good | **Excellent** |
| **OOV Handling** | ❌ No | ❌ No | ❌ No | **✅ Yes (computes from n-grams)** |
| **Semantic & Syntactic** | Good | Strong | Strong | **Best on morphology** |

---

## 💻 Python Example (Training with Gensim)

```python
from gensim.models import Word2Vec, FastText

# Sample tokenized sentences
sentences = [
    ["machine", "learning", "is", "fascinating"],
    ["deep", "learning", "powers", "modern", "artificial", "intelligence"],
    ["natural", "language", "processing", "uses", "word", "embeddings"]
]

# 1. Train Word2Vec (Skip-Gram: sg=1, CBOW: sg=0)
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=10)
vector_w2v = w2v_model.wv['learning']

# 2. Train FastText (Supports OOV words via subword n-grams)
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, min_n=3, max_n=6, epochs=10)
# 'learnings' is not in training vocab, but FastText can construct its vector!
vector_ft_oov = ft_model.wv['learnings']

print("Word2Vec vector shape:", vector_w2v.shape)
print("FastText OOV vector shape:", vector_ft_oov.shape)
```
>>>>>>> 0fbc6b0b6a6b3b5e9cfb8d4fcfd91eea1d2fab86
