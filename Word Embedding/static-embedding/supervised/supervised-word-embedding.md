# Supervised Word Embeddings

Supervised word embeddings are task-specific, dense vector representations learned end-to-end as part of a supervised neural network (such as a sentiment classifier, spam detector, or document categorizer).

Unlike self-supervised methods (e.g., Word2Vec) that learn general semantic relationships from raw text, supervised embeddings are optimized directly to minimize the loss for a specific target task.

---

## 🧠 Core Concept & Intuition

1. **Vocabulary Size ($V$):** Define the total number of unique words in our dictionary (e.g., $V = 500$ or $V = 50,000$).
2. **Embedding Dimension ($d$):** Choose how many features (dimensions) represent each word (e.g., $d = 5, 100, 300$).
3. **Lookup Table (Weight Matrix):** An embedding layer is simply a weight matrix $W_E$ of size $V \times d$.
4. **Random Initialization:** Initially, all word vectors contain small random weights.
5. **Backpropagation & Optimization:** As the neural network trains on labeled data (e.g., predicting sentiment $y \in \{0, 1\}$), gradients flow back into the embedding matrix, updating the vectors so that words informative for the task cluster together.

```
Raw Text: "movie is good"
       │
       ▼ [Tokenization & Indexing]
Token IDs: [12, 4, 88]
       │
       ▼ [Embedding Lookup Layer: V × d]
Dense Vectors: [ [0.25, -0.11, ...], [0.03, 0.44, ...], [0.89, 0.62, ...] ]
       │
       ▼ [Pooling / Flatten / Hidden Layers]
Hidden Representation
       │
       ▼ [Output Layer + Activation]
Prediction: ŷ = 0.94 (Positive Sentiment)
       │
       ▼ [Loss Function: Binary Cross-Entropy (ŷ vs. y)]
Backpropagation ➔ Updates Embedding Matrix Weights!
```

---

## 🧮 Understanding Neural Network Parameters

When we hear that a model has **8 Billion parameters** (like Llama 3 8B) or a small custom network has **2,650 parameters**, what does that mean?

> **Parameters** are all the trainable weights ($W$) and biases ($b$) across all layers in the neural network that are adjusted during gradient descent.

---

### Step-by-Step Parameter Calculation Example

Let's design a simple supervised classification network:
- **Vocabulary Size ($V$):** 500 words
- **Embedding Dimension ($d$):** 5 features per word
- **Aggregation:** Average pooling over tokens $\to$ produces a single vector of shape $(1, 5)$
- **Hidden Layer 1:** 10 neurons (Dense layer + ReLU)
- **Hidden Layer 2:** 10 neurons (Dense layer + ReLU)
- **Output Layer:** 1 neuron (Dense layer + Sigmoid for binary output)

```
Input Tokens (IDs)
        │
┌───────▼───────────────────────────────────────────────┐
│ 1. Embedding Layer: V × d (500 × 5)                   │ ➔ 2,500 Weights (No Bias)
└───────┬───────────────────────────────────────────────┘
        │ Embedding Output: (1 × 5)
┌───────▼───────────────────────────────────────────────┐
│ 2. Hidden Layer 1: Dense(5 ➔ 10)                      │ ➔ (5 × 10) Weights + 10 Biases = 60
└───────┬───────────────────────────────────────────────┘
        │ Layer 1 Output: (1 × 10)
┌───────▼───────────────────────────────────────────────┐
│ 3. Hidden Layer 2: Dense(10 ➔ 10)                     │ ➔ (10 × 10) Weights + 10 Biases = 110
└───────┬───────────────────────────────────────────────┘
        │ Layer 2 Output: (1 × 10)
┌───────▼───────────────────────────────────────────────┐
│ 4. Output Layer: Dense(10 ➔ 1)                        │ ➔ (10 × 1) Weights + 1 Bias = 11
└───────┬───────────────────────────────────────────────┘
        │
        ▼
   Prediction ŷ (0 or 1)
```

#### Detailed Breakdown:

| Layer | Computation | Weights Formula | Biases Formula | Total Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding Layer** | Table lookup | $V \times d = 500 \times 5$ | None | **2,500** |
| **Hidden Layer 1** | $W_1 x + b_1$ | $5 \times 10$ | $10$ | **60** |
| **Hidden Layer 2** | $W_2 h_1 + b_2$ | $10 \times 10$ | $10$ | **110** |
| **Output Layer** | $W_3 h_2 + b_3$ | $10 \times 1$ | $1$ | **11** |
| **Total Model Parameters** | | | | **2,681 Parameters** |

> **Key Rule:** For a dense (linear) layer connecting $N_{in}$ inputs to $N_{out}$ neurons:
> $$\text{Parameters} = (N_{in} \times N_{out}) + N_{out}$$

---

## 💻 Code Implementation

### 1. PyTorch Implementation

```python
import torch
import torch.nn as nn

class SupervisedEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size=500, embed_dim=5, hidden_dim=10, num_classes=1):
        super().__init__()
        # Embedding Layer: Shape (500, 5)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        # Dense Layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)         # Shape: (batch_size, seq_len, embed_dim)
        pooled = embeds.mean(dim=1)        # Average pooling across sequence length -> (batch_size, embed_dim)
        
        h1 = self.relu1(self.fc1(pooled))  # Shape: (batch_size, hidden_dim)
        h2 = self.relu2(self.fc2(h1))      # Shape: (batch_size, hidden_dim)
        output = self.sigmoid(self.out(h2))# Shape: (batch_size, 1)
        return output

# Count trainable parameters
model = SupervisedEmbeddingClassifier()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}") # Output: 2681
```

### 2. TensorFlow / Keras Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    # Embedding Layer
    layers.Embedding(input_dim=500, output_dim=5, name="embedding_layer"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(10, activation='relu', name="hidden_layer_1"),
    layers.Dense(10, activation='relu', name="hidden_layer_2"),
    layers.Dense(1, activation='sigmoid', name="output_layer")
])

model.summary()
```

---

## ⚖️ Advantages vs. Disadvantages

### Advantages
- **Task-Tailored Semantics:** Embeddings adapt precisely to the nuances of the specific problem (e.g., in movie sentiment, "unpredictable" may be mapped as positive, whereas in financial forecasting it may be negative).
- **End-to-End Optimization:** No need to precompute or manage external embedding files.

### Disadvantages
- **Data Hungry:** Requires a large labeled dataset to learn good representations; otherwise, the embedding matrix severely overfits.
- **Out-of-Vocabulary (OOV) Issue:** Any word not present in the fixed training vocabulary cannot be represented during inference.
- **Non-transferable:** Embeddings learned for spam detection rarely transfer well to biomedical classification.
