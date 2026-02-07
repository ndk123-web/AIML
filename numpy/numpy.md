# NumPy - Numerical Python

## Overview
NumPy is a Python library for numerical computing that provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

## Key Features
- **N-dimensional arrays (ndarray)**: Efficient storage and manipulation
- **Broadcasting**: Automatic element-wise operations on arrays of different shapes
- **Mathematical functions**: Linear algebra, Fourier transforms, random number generation
- **Performance**: Written in C with Python interface, much faster than pure Python
- **Integration**: Works well with pandas, matplotlib, scikit-learn, and other libraries

## Installation
```python
pip install numpy
```

## Basic Concepts

### 1. Arrays (ndarray)
NumPy's core data structure is the ndarray (N-dimensional array).

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4])  # 1D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array

# Different ways to create arrays
zeros = np.zeros((3, 4))  # All zeros
ones = np.ones((2, 3))  # All ones
arange = np.arange(0, 10, 2)  # Like Python range
linspace = np.linspace(0, 1, 5)  # Evenly spaced values
random = np.random.rand(3, 3)  # Random values
identity = np.eye(3)  # Identity matrix
```

### 2. Array Attributes
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape      # Dimensions: (2, 3)
arr.dtype      # Data type: int64
arr.size       # Total number of elements: 6
arr.ndim       # Number of dimensions: 2
arr.itemsize   # Size of each element in bytes
arr.nbytes     # Total bytes consumed: 48
```

### 3. Data Types
```python
# Specify dtype when creating arrays
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1.5, 2.5, 3.5], dtype=np.float64)
arr_bool = np.array([True, False, True], dtype=bool)
arr_complex = np.array([1+2j, 3+4j], dtype=np.complex128)
```

### 4. Indexing and Slicing
```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arr[0]        # First element: 0
arr[-1]       # Last element: 9
arr[2:5]      # Elements from index 2 to 4: [2, 3, 4]
arr[::2]      # Every second element: [0, 2, 4, 6, 8]
arr[::-1]     # Reversed: [9, 8, 7, ..., 0]

# 2D arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[0]      # First row: [1, 2, 3]
arr2d[0, 1]   # Element at row 0, col 1: 2
arr2d[:, 1]   # Second column: [2, 5, 8]
arr2d[1:, ::2]  # Complex slicing
```

### 5. Reshaping Arrays
```python
arr = np.arange(12)
reshaped = arr.reshape(3, 4)  # Change shape to (3, 4)
flattened = reshaped.flatten()  # Flatten to 1D
raveled = reshaped.ravel()  # Also flatten

# Transpose
transposed = arr2d.T
```

### 6. Basic Operations
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Element-wise operations
addition = arr1 + arr2  # [5, 7, 9]
subtraction = arr1 - arr2  # [-3, -3, -3]
multiplication = arr1 * arr2  # [4, 10, 18]
division = arr1 / arr2  # [0.25, 0.4, 0.5]
power = arr1 ** 2  # [1, 4, 9]

# Scalar operations
arr1 + 10  # [11, 12, 13]
arr1 * 2   # [2, 4, 6]
```

### 7. Broadcasting
Broadcasting allows operations on arrays of different shapes.

```python
arr1 = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
arr2 = np.array([10, 20, 30])  # Shape: (3,)

result = arr1 + arr2  # arr2 is broadcast to match arr1's shape
# [[11, 22, 33],
#  [14, 25, 36]]
```

### 8. Array Functions

#### Aggregate Functions
```python
arr = np.array([1, 2, 3, 4, 5])

np.sum(arr)        # Sum: 15
np.mean(arr)       # Mean: 3.0
np.median(arr)     # Median: 3.0
np.std(arr)        # Standard deviation
np.var(arr)        # Variance
np.min(arr)        # Minimum: 1
np.max(arr)        # Maximum: 5
np.prod(arr)       # Product: 120

# Along axis for 2D arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
np.sum(arr2d, axis=0)  # Sum along rows: [5, 7, 9]
np.sum(arr2d, axis=1)  # Sum along columns: [6, 15]
```

#### Mathematical Functions
```python
arr = np.array([0, 1, 2, 3, 4])

np.sqrt(arr)       # Square root
np.exp(arr)        # Exponential
np.log(arr[1:])    # Natural logarithm
np.sin(arr)        # Sine
np.cos(arr)        # Cosine
np.abs(arr)        # Absolute value
np.ceil(3.2)       # Ceiling: 4
np.floor(3.7)      # Floor: 3
np.round(3.7)      # Round: 4
```

### 9. Sorting and Searching
```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

np.sort(arr)       # Sorted array: [1, 1, 2, 3, 4, 5, 6, 9]
np.argsort(arr)    # Indices that would sort: [1, 3, 6, 0, 2, 4, 7, 5]
np.argmax(arr)     # Index of max value: 5
np.argmin(arr)     # Index of min value: 1
np.where(arr > 3)  # Indices where condition is true
```

### 10. Stacking and Splitting
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stacking
np.concatenate([arr1, arr2])  # [1, 2, 3, 4, 5, 6]
np.vstack([arr1, arr2])       # Vertical: [[1, 2, 3], [4, 5, 6]]
np.hstack([arr1, arr2])       # Horizontal: [1, 2, 3, 4, 5, 6]

# Splitting
arr = np.arange(9)
np.split(arr, 3)  # Split into 3 equal parts
np.array_split(arr, 4)  # Split into 4 parts (unequal divisions allowed)
```

### 11. Linear Algebra
```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)  # Matrix multiplication
C = A @ B  # Also matrix multiplication (Python 3.5+)

# Determinant
det = np.linalg.det(A)

# Inverse
A_inv = np.linalg.inv(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solving linear equations: Ax = b
x = np.linalg.solve(A, np.array([1, 2]))
```

### 12. Random Numbers
```python
# Generate random numbers
np.random.rand(3, 4)       # Uniform [0, 1)
np.random.randn(3, 4)      # Normal distribution
np.random.randint(0, 10, 5)  # Random integers
np.random.choice([1, 2, 3, 4, 5], 3)  # Random choice

# Set seed for reproducibility
np.random.seed(42)
```

### 13. Boolean Indexing and Masking
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Boolean indexing
mask = arr > 5
arr[mask]  # [6, 7, 8, 9, 10]

# Multiple conditions
mask2 = (arr > 3) & (arr < 8)
arr[mask2]  # [4, 5, 6, 7]

# Using np.where for conditional replacement
result = np.where(arr > 5, arr * 2, arr)
```

### 14. Copying Arrays
```python
arr = np.array([1, 2, 3])

# Reference (same object)
arr_ref = arr
arr_ref[0] = 999  # Changes original!

# Shallow copy
arr_copy = arr.copy()
arr_copy[0] = 999  # Doesn't affect original

# View
arr_view = arr[:]  # Changes affect original
```

## Important Tips for Machine Learning

1. **Vectorization**: Always try to use NumPy operations instead of loops for better performance
2. **Memory efficiency**: Use appropriate dtypes (float32 vs float64) based on requirements
3. **Broadcasting**: Master this concept for working with different shaped arrays
4. **Avoid copies**: Be aware of when you're creating copies vs views
5. **Random seeds**: Set seeds for reproducible results in ML experiments
6. **Normalization**: Use NumPy to normalize features in ML pipelines
7. **Batch operations**: Use NumPy for efficient batch processing

## Common Use Cases in ML

```python
# Feature normalization
X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
indices = np.random.permutation(len(X))
train_idx = indices[:int(0.8*len(X))]
test_idx = indices[int(0.8*len(X)):]

# Computing distances (Euclidean)
distances = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))
```

## Resources
- Official NumPy Documentation: https://numpy.org/doc/
- NumPy Tutorials: https://numpy.org/learn/
- Practice with NumPy: https://github.com/rougier/numpy-100

## Summary
NumPy is essential for:
- Efficient numerical computing
- Machine learning and data science
- Scientific computing
- Data manipulation and analysis

Master NumPy to become proficient in Python-based data science!
