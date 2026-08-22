# Contextualized Word Embeddings

Contextualized word embeddings generate dynamic vector representations of words based on their surrounding context within a sequence, completely resolving the fundamental polysemy limitation of static word embeddings.

---

## 🛑 The Core Problem with Static Embeddings: Polysemy

In static embedding models (such as Word2Vec, GloVe, and FastText), each word has a **single, fixed vector** in memory regardless of how it is used.

```text
"bank" ➔ [0.21, 0.54, -0.12, 0.88, ...] (Fixed Vector)
```

### The Polysemy Conflict

Consider these two sentences:
1. *"I deposited money in the **bank**."* $\to$ Financial institution
2. *"I sat near the river **bank**."* $\to$ Sloping land beside water

```
Static Embeddings:
"bank" (Sentence 1: Financial) ──┐
                                 ├──▶ SAME Vector: [0.21, 0.54, -0.12, ...]
"bank" (Sentence 2: River)     ──┘
❌ Loss of contextual meaning!
```

Because language is inherently polysemous (words have multiple meanings depending on context), static vectors are forced to collapse distinct semantic concepts into an averaged compromises vector.

---

## 🌟 The Solution: Contextualized Word Embeddings

Contextualized embeddings compute representations dynamically:

$$\text{Embedding}(\text{word}) = f(\text{word}, \text{surrounding sentence context})$$

```
Contextualized Embeddings (BERT / Transformers):

"I deposited money in the bank"  ──▶ BERT ──▶ E_financial("bank") = [ 0.82, -0.41,  0.19, ...]
"I sat near the river bank"      ──▶ BERT ──▶ E_river("bank")     = [-0.15,  0.73, -0.66, ...]
✅ Accurate, context-aware representations!
```

---

## 🏛️ Architectural Evolution: How Context is Built

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Static Embeddings (Word2Vec, GloVe) [2013-2014]                          │
│    Lookup table: Input ID ➔ Fixed Vector (0 sequence awareness)             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Feature-based Sequential Context: ELMo [Peters et al., 2018]             │
│    Bidirectional LSTM: Concatenates forward LSTM + backward LSTM states     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Deep Bidirectional Self-Attention: BERT [Devlin et al., 2018]           │
│    Transformer Encoder: All tokens attend to all tokens simultaneously      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Deep Dive: How BERT Generates Contextual Embeddings

**BERT (Bidirectional Encoder Representations from Transformers)** processes tokens using Multi-Head Self-Attention across multiple encoder layers.

```
Input Sentence: "The bank approved my loan"
      │
      ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Token: "The" │ │ Token:"bank" │ │Token:"approv"│ │ Token: "my"  │ │ Token:"loan" │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │                │
       ▼                ▼                ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Transformer Encoder Layer 1 (Self-Attention)                   │
│  "bank" computes attention scores with "approved", "loan", "The", and "my"      │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  Transformer Encoder Layer 12 (Final Layer)                      │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                                       ▼
   Dynamic Contextualized Vector for "bank" (Weighted heavily by "loan" & "approved")
```

---

## 🎯 Case Study: Named Entity Recognition (NER)

Contextual representations are critical for token-level tasks like Named Entity Recognition (NER):

### Example 1:
> *"**Apple** announced the new iPhone."*
- BERT attends to `"iPhone"` and `"announced"` $\to$ predicts entity label **`B-ORG`** (Organization).

### Example 2:
> *"I ate a crisp **apple** for breakfast."*
- BERT attends to `"ate"`, `"crisp"`, and `"breakfast"` $\to$ predicts entity label **`O`** (Outside / Not an entity).

---

## 📊 Static vs. Contextualized Embeddings Comparison

| Feature | Static Embedding (Word2Vec, GloVe) | Contextualized Embedding (BERT, RoBERTa) |
| :--- | :--- | :--- |
| **Representation Nature** | Fixed global lookup table | Dynamic function of the whole sequence |
| **Polysemy Handling** | ❌ Fails (single representation) | ✅ Handles polysemy seamlessly |
| **Mechanism** | Hash / Matrix Table Lookup ($\mathcal{O}(1)$) | Multi-Layer Transformer Self-Attention ($\mathcal{O}(N^2)$) |
| **Computation Cost** | Extremely low inference cost | Higher compute & GPU memory requirements |
| **Out-Of-Vocabulary** | Fails on unseen words | Handled via Subword Tokenizers (WordPiece/BPE) |
| **Transfer Learning** | Static initialization for downstream models | Fine-tuned end-to-end for downstream tasks |

---

## 💻 Python Example: Extracting Contextual Embeddings with Hugging Face

```python
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Load Pretrained Model & Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 2. Sentences with polysemous word 'bank'
sentences = [
    "I deposited money in the bank account.",
    "The river bank was muddy and overflowing."
]

# 3. Tokenize & Forward Pass
inputs = tokenizer(sentences, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 4. Extract token embeddings from last hidden state (Shape: batch_size, seq_len, 768)
last_hidden_states = outputs.last_hidden_state

# Find index of token 'bank' in both sentences
bank_token_id = tokenizer.convert_tokens_to_ids("bank")
idx_sent1 = (inputs['input_ids'][0] == bank_token_id).nonzero().item()
idx_sent2 = (inputs['input_ids'][1] == bank_token_id).nonzero().item()

emb_bank_financial = last_hidden_states[0, idx_sent1]
emb_bank_river = last_hidden_states[1, idx_sent2]

# Compute Cosine Similarity between the two 'bank' embeddings
cos_sim = torch.nn.functional.cosine_similarity(emb_bank_financial, emb_bank_river, dim=0)
print(f"Cosine Similarity between 'bank' in financial vs river context: {cos_sim.item():.4f}")
# Output will be significantly lower than 1.0 (e.g. ~0.55), proving context differentiation!
```
