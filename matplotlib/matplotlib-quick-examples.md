# Matplotlib Quick Examples

This notebook provides quick examples for common Matplotlib plots.

---

## 1. Line Plot
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

## 2. Scatter Plot
```python
plt.scatter([1, 2, 3], [4, 5, 6], color='red')
plt.title('Scatter Plot')
plt.show()
```

## 3. Bar Chart
```python
plt.bar(['A', 'B', 'C'], [10, 20, 15], color='green')
plt.title('Bar Chart')
plt.show()
```

## 4. Histogram
```python
import numpy as np
data = np.random.randn(1000)
plt.hist(data, bins=30, color='purple')
plt.title('Histogram')
plt.show()
```

## 5. Pie Chart
```python
sizes = [30, 20, 50]
labels = ['A', 'B', 'C']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()
```

---

For more, see [matplotlib-quick-learn.md](matplotlib-quick-learn.md)
