# Transformer

## Overview

- It Has 2 Important Components
  1. Encoder
     - Simple to generate contextualize embedding for each word in a sentence
     - Example: BERT

  2. Decoder
     - Simple to predict the next word that's it
  - Example: GPT 2.2

## How ?

**1. Encoder (The Understander)**

- **Tokenization:** Converts the input text into Tokens.
- **Static Embedding:** Maps those Tokens to Static Embeddings (e.g., 768 dimensions).
- **Positional Encoding:** Adds position numbers to the embeddings so the model knows the order of the words.
- **Self-Attention ($Q, K, V$):**
  - Creates Query ($Q$), Key ($K$), and Value ($V$) from its **own input sentence**.
  - Multiplies $Q \cdot K$ to see how each word relates to every other word in the sentence.
  - Merges this score with $V$ to create a new, context-aware vector for each word.
  - Repeats this process for $N$ layers (e.g., 12 layers).
- **Final Output:** Sends the completely **Contextualized Embeddings** of the input sentence to the Decoder.

---

**2. Decoder (The Generator)**

- **Task:** Generates the output text one word at a time.
- **Two Types of Attention ($Q, K, V$):**
  - **i. Masked Self-Attention (Self-Check):**
    - Looks only at the words the Decoder has _already generated_.
    - $Q, K, V$ all come from the **Decoder's own generated words**.
    - It is "Masked" so it cannot cheat and look at future words.
  - **ii. Cross-Attention (Ask the Teacher):**
    - Connects the Decoder with the Encoder.
    - **$Q$ (Query)** comes from the **Decoder** ("What should I focus on next?").
    - **$K, V$ (Key, Value)** come from the **Encoder's** final contextualized embeddings ("Here is the context of the original sentence").
- **Final Embedding:** Gives the final contextual embedding for the exact word it needs to generate right now.
- **MLP / Linear Layer:** Passes this embedding through a dense layer that matches your total **Vocabulary Size**.
- **Softmax (Prediction):** Converts the output into probabilities. The word with the **highest probability** is selected as the next predicted word!
