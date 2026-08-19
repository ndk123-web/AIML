# Word Embeddings — Complete Guide & Taxonomy

Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to proximate points. They form the foundational building blocks for modern Natural Language Processing (NLP) and Large Language Models (LLMs).

---

## 🗺️ Taxonomy of Word Embeddings

```
Word Embeddings
│
├── 1. Static Embeddings (One fixed vector per word)
│   ├── Supervised Embeddings (Task-specific, learned end-to-end)
│   │   └── Embedding Layer (e.g., PyTorch nn.Embedding, Keras Embedding)
│   │
│   └── Self-Supervised Embeddings (Pretrained on large unlabelled text corpora)
│       ├── Prediction-Based (Local Window)
│       │   ├── Word2Vec — CBOW (Predicts target word from context)
│       │   └── Word2Vec — Skip-Gram (Predicts context from target word)
│       ├── Matrix Factorization (Global Statistics)
│       │   └── GloVe (Global Vectors from co-occurrence matrix)
│       └── Subword-Based (Character n-grams for morphology & OOV)
│           └── FastText
│
└── 2. Contextualized Embeddings (Dynamic vector based on sentence context)
    ├── Recurrent-Based: ELMo (Bidirectional LSTM)
    ├── Encoder-Based: BERT, RoBERTa, DeBERTa (Bidirectional Transformer)
    └── Autoregressive / Generative: GPT series (Causal Transformer)
```

---

## 📂 Directory Structure & Modules

| Module | Description | Path |
| :--- | :--- | :--- |
| **Supervised Static Embeddings** | Learning task-specific lookup tables from labeled data, embedding parameter math, and neural network parameter breakdowns. | [`static-embedding/supervised/supervised-word-embedding.md`](./static-embedding/supervised/supervised-word-embedding.md) |
| **Self-Supervised Static Embeddings** | Learning general-purpose representations from unlabeled text using CBOW, Skip-Gram (Word2Vec), GloVe, and FastText. | [`static-embedding/self-supervised/self-supervised-word-embedding.md`](./static-embedding/self-supervised/self-supervised-word-embedding.md) |
| **Contextualized Embeddings** | Overcoming polysemy and static limitations via dynamic, attention-driven representations (BERT, Transformers). | [`contextualize-embedding/context.md`](./contextualize-embedding/context.md) |

---

## ⚖️ Static vs. Contextualized Embeddings at a Glance

| Feature | Static Embeddings (Word2Vec, GloVe) | Contextualized Embeddings (BERT, GPT) |
| :--- | :--- | :--- |
| **Vector Assignment** | 1 word = 1 fixed vector | 1 word = dynamic vector dependent on sentence |
| **Polysemy Support** | ❌ Fails (e.g., "river bank" vs "money bank") | ✅ Supported (different vector for each meaning) |
| **Compute Overhead** | Very Fast (Simple $\mathcal{O}(1)$ table lookup) | Heavy (Multi-head self-attention layers) |
| **OOV Handling** | Poor (unless using subword like FastText) | Excellent (Subword tokenizers: BPE, WordPiece) |
| **Downstream Transfer** | Used as static feature inputs | Fine-tuned end-to-end or used as feature extractors |

---

## 🎯 Learning Path

1. **Start with [Supervised Word Embeddings](./static-embedding/supervised/supervised-word-embedding.md)** to understand how embedding lookup layers function inside neural networks and how parameters are calculated.
2. **Explore [Self-Supervised Word Embeddings](./static-embedding/self-supervised/self-supervised-word-embedding.md)** to master Word2Vec (CBOW & Skip-Gram), GloVe, and FastText.
3. **Advance to [Contextualized Word Embeddings](./contextualize-embedding/context.md)** to understand how modern Transformers solve polysemy and generate dynamic contextual representations.
