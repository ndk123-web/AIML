# Matplotlib Quick Learn Guide

Matplotlib is a powerful Python library for creating static, animated, and interactive visualizations.

## Installation
```bash
pip install matplotlib
```

## Basic Example
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, marker='o')
plt.title('Simple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

## Common Plot Types

### 1. Line Plot
```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

### 2. Scatter Plot
```python
plt.scatter([1, 2, 3], [4, 5, 6])
plt.show()
```

### 3. Bar Chart
```python
plt.bar(['A', 'B', 'C'], [10, 20, 15])
plt.show()
```

### 4. Histogram
```python
import numpy as np
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.show()
```

### 5. Pie Chart
```python
sizes = [30, 20, 50]
labels = ['A', 'B', 'C']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.show()
```

## Customization Tips
- Use `plt.title()`, `plt.xlabel()`, `plt.ylabel()` for labels
- Change colors and styles with arguments like `color`, `linestyle`, `marker`
- Save figures with `plt.savefig('filename.png')`

## Resources
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Gallery of Examples](https://matplotlib.org/stable/gallery/index.html)
