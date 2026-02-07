# NumPy Quick Reference Guide

## Array Creation
```python
np.array([1, 2, 3])           # From list
np.zeros((3, 4))              # All zeros
np.ones((3, 4))               # All ones
np.arange(0, 10, 2)           # 0, 2, 4, 6, 8
np.linspace(0, 1, 5)          # 5 evenly spaced from 0 to 1
np.eye(3)                     # Identity matrix
np.random.rand(3, 4)          # Random [0, 1)
np.full((3, 4), 5)            # All 5s
```

## Array Properties
```python
arr.shape                     # Dimensions
arr.dtype                     # Data type
arr.size                      # Total elements
arr.ndim                      # Number of dimensions
arr.T                         # Transpose
```

## Indexing & Slicing
```python
arr[0]                        # First element
arr[-1]                       # Last element
arr[2:5]                      # Index 2 to 4
arr[::2]                      # Every second element
arr2d[0, 1]                   # Row 0, Column 1
arr2d[:, 1]                   # All rows, Column 1
arr2d[1:, ::2]                # Complex slicing
```

## Reshaping
```python
arr.reshape(3, 4)             # Change shape to (3, 4)
arr.flatten()                 # Convert to 1D
arr.ravel()                   # Also convert to 1D
arr.T                         # Transpose
```

## Basic Operations
```python
arr1 + arr2                   # Element-wise addition
arr1 - arr2                   # Element-wise subtraction
arr1 * arr2                   # Element-wise multiplication
arr1 / arr2                   # Element-wise division
arr1 ** 2                     # Element-wise power
np.sqrt(arr)                  # Square root
np.exp(arr)                   # Exponential
```

## Broadcasting
```python
arr1 = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
arr2 = np.array([10, 20, 30])             # Shape (3,)
result = arr1 + arr2                      # arr2 broadcasts to (2, 3)
```

## Aggregate Functions
```python
np.sum(arr)                   # Sum of all elements
np.mean(arr)                  # Mean
np.median(arr)                # Median
np.std(arr)                   # Standard deviation
np.var(arr)                   # Variance
np.min(arr)                   # Minimum
np.max(arr)                   # Maximum
np.prod(arr)                  # Product

# With axis parameter (for 2D arrays)
np.sum(arr, axis=0)           # Sum along rows
np.sum(arr, axis=1)           # Sum along columns
```

## Sorting & Searching
```python
np.sort(arr)                  # Sorted array
np.argsort(arr)               # Indices that would sort
np.argmax(arr)                # Index of max value
np.argmin(arr)                # Index of min value
np.where(arr > 5)             # Indices where condition true
```

## Boolean Indexing
```python
mask = arr > 5                # Create boolean mask
arr[mask]                     # Select elements where True
arr[(arr > 3) & (arr < 8)]    # Multiple conditions
np.where(arr > 5, arr*2, arr) # Conditional replacement
```

## Stacking & Splitting
```python
np.concatenate([arr1, arr2])  # Concatenate arrays
np.vstack([arr1, arr2])       # Stack vertically
np.hstack([arr1, arr2])       # Stack horizontally
np.split(arr, 3)              # Split into 3 parts
```

## Linear Algebra
```python
np.dot(A, B)                  # Matrix multiplication
A @ B                         # Also matrix multiplication
np.linalg.det(A)              # Determinant
np.linalg.inv(A)              # Inverse
np.linalg.eig(A)              # Eigenvalues and eigenvectors
np.linalg.solve(A, b)         # Solve linear equations Ax=b
```

## Random Numbers
```python
np.random.rand(3, 4)          # Uniform [0, 1)
np.random.randn(3, 4)         # Normal distribution
np.random.randint(0, 10, 5)   # Random integers 0-9
np.random.choice([1,2,3], 5)  # Random choice
np.random.shuffle(arr)        # Shuffle in-place
np.random.seed(42)            # Set seed for reproducibility
```

## Useful Utilities
```python
np.unique(arr)                # Unique elements
np.bincount(arr)              # Count occurrences
np.cumsum(arr)                # Cumulative sum
np.cumprod(arr)               # Cumulative product
np.diff(arr)                  # Differences between elements
```

## Type Conversion
```python
arr.astype(np.float32)        # Convert to float32
arr.astype(int)               # Convert to integer
arr.astype(bool)              # Convert to boolean
```

## Memory & Copying
```python
arr.copy()                    # Create a copy
arr[:]                        # Create a view (not a copy)
arr.flags['WRITEABLE']        # Check if writeable
```

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Shape mismatch | Arrays have incompatible shapes | Use broadcasting or reshape |
| Index out of bounds | Index exceeds array size | Check array size and indexing |
| Type error | Incompatible data types | Convert using astype() |

## Performance Tips
1. **Vectorize**: Avoid loops, use NumPy operations
2. **Use appropriate dtypes**: float32 < float64 (memory-wise)
3. **In-place operations**: Use `+=` instead of `=` to save memory
4. **Avoid copies**: Be aware of when you're creating copies vs views
5. **Use broadcasting**: Leverage it for efficient operations

## Cheat Sheet Commands
```python
import numpy as np

# Check version
print(np.__version__)

# Get documentation
help(np.array)
np.info(np.array)

# List all functions
dir(np)

# Check array info
np.info(arr)
```
