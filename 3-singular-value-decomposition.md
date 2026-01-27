# Singular Value Decomposition (SVD)

## What

Singular Value Decomposition (SVD) is a matrix factorization technique that breaks a matrix into three simpler matrices:

```
A = U · Σ · Vᵀ
```

- **U**: left singular vectors (output directions)
- **Σ (Sigma)**: singular values (importance of information)
- **Vᵀ**: right singular vectors (input directions)

---

## Why

Real-world data is often noisy, high-dimensional, and redundant.
SVD helps to:

- identify the most important patterns in data
- reduce dimensionality
- remove noise
- compress data efficiently

---

## How

SVD decomposes a matrix into rotations and scaling:

1. Rotate the data into meaningful directions
2. Scale data by importance
3. Rotate into final output space

In practice, SVD is computed using numerical libraries (e.g., NumPy), not manually.

---

## Example

For a data matrix representing users and movies:

- rows → users
- columns → movies

SVD extracts:

- hidden user preferences
- hidden movie characteristics

This idea is widely used in recommendation systems.

---

## Logic

Not all dimensions in data are equally important.
SVD keeps strong patterns (large singular values) and discards weak ones (small singular values).

---

## Concept

**SVD reveals the core structure of a matrix by separating signal from noise.**

---

## Applications

- Dimensionality reduction (PCA uses SVD internally)
- Image compression
- Noise reduction
- Recommendation systems
- Text analysis (Latent Semantic Analysis)

---

## Summary

- SVD factorizes a matrix into three parts
- Singular values represent information strength
- Keeping top singular values preserves essential data
- Widely used in machine learning and data processing

---

### One-line takeaway

> **SVD simplifies complex data while preserving what matters most.**
