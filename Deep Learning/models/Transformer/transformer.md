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

## Why Normalization?

- While **Kaiming** and **Xavier** weight initializations help prevent exploding/vanishing gradients at the start, continuous weight updates during training can still cause layer outputs to become unstable (too huge or too small).
- We use normalization to balance these outputs dynamically during training.
- **Standard Flow:** `Weighted Sum -> Normalization -> Activation`
- Ultimately, normalization ensures stable, faster training and acts as a constant safeguard against gradient problems.

## Types of Normalization

### 1. Batch Normalization (Batch Norm)
- **Best For:** Convolutional Neural Networks (CNNs).
- **Mechanism:** Normalizes **vertically** (feature-wise, across the entire batch).

**A. During Training:**
- Calculates the mean ($\mu_{\text{batch}}$) and variance ($\sigma_{\text{batch}}^2$) directly from the **current mini-batch**.
- **Formula:** 
  $$ y_i = \gamma \left( \frac{x_i - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}} \right) + \beta $$
- **Running Stats & Momentum:** During training, it also calculates a **global** mean and variance (`running_mean`, `running_var`) using an Exponential Moving Average. 
  - The **momentum** controls how much the current batch updates this global average.
  - $\text{running\_mean} = (1 - \text{momentum}) \times \text{running\_mean} + \text{momentum} \times \mu_{\text{batch}}$

**B. During Prediction (Inference/Testing):**
- In testing, your batch size might just be 1. Calculating a mean of a single image doesn't make sense!
- So, it completely ignores the batch statistics and uses the **running statistics** gathered during training.
- **Formula:** 
  $$ y_i = \gamma \left( \frac{x_i - \text{running\_mean}}{\sqrt{\text{running\_var} + \epsilon}} \right) + \beta $$

**Key Variables:**
- $\gamma$ (Scale) & $\beta$ (Shift): Learnable parameters optimized during training.
- $\epsilon$: A tiny constant (e.g., $10^{-5}$) to prevent division by zero.

### 2. Layer Normalization (Layer Norm)
- **Best For:** Transformers and RNNs.
- **Mechanism:** Normalizes **horizontally** (sample-wise, across all features for a single input).
- **Formula:**
  $$ y_i = \gamma \left( \frac{x_i - \mu_{\text{sample}}}{\sqrt{\sigma_{\text{sample}}^2 + \epsilon}} \right) + \beta $$
  - $x_i$: The input vector (weighted sum) for one sample.
  - $\mu_{\text{sample}}$: Mean of all features for that specific sample.
  - $\sigma_{\text{sample}}^2$: Variance of all features for that specific sample.
  - $\gamma$ & $\beta$: Learnable parameters optimized during training to scale and shift the output.
