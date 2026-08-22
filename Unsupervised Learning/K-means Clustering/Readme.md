# K-Means Clustering

K-Means is an unsupervised learning algorithm used to group similar data points into k clusters. It's one of the most popular clustering algorithms due to its simplicity and efficiency.

## What is K-Means?

K-Means clustering partitions data into k distinct clusters by minimizing the variance within each cluster. The algorithm works by:

1. **Initialize**: Randomly select k points as initial centroids
2. **Assign**: Assign each point to the nearest centroid
3. **Update**: Calculate new centroids as the mean of all points in each cluster
4. **Repeat**: Steps 2-3 until centroids don't change or maximum iterations reached

## Key Concepts

### Centroid
- The center point of a cluster
- Calculated as the mean of all points in that cluster

### Inertia (Within-cluster sum of squares)
- Sum of squared distances from each point to its centroid
- Lower inertia = tighter, more compact clusters

### Distance Metric
- Typically uses Euclidean distance to find nearest centroid

## Finding Optimal K - Elbow Method

The **Elbow Method** helps determine the best number of clusters:

1. Run K-Means for different values of k (e.g., 1 to 10)
2. Calculate inertia for each k
3. Plot k vs inertia
4. Look for the "elbow" point where inertia decrease slows down
5. The k at the elbow is the optimal number

This method balances model complexity with cluster quality.

## Feature Scaling

When features have different scales (like Age and Income), it's important to **normalize** them:

- **MinMaxScaler**: Scales features to range [0, 1]
- **StandardScaler**: Standardizes features (mean=0, std=1)

Without scaling, features with larger ranges dominate the distance calculation, leading to biased results.

## Implementation Steps

1. **Load Data**: Import your dataset
2. **Explore**: Visualize data to understand distribution
3. **Scale Features**: Normalize features if they have different units
4. **Apply K-Means**: Fit the algorithm with chosen k
5. **Visualize**: Plot clusters to verify grouping
6. **Evaluate**: Assess cluster quality

## Scikit-Learn Usage

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Create and fit model
kmeans = KMeans(n_clusters=k, random_state=42)
predictions = kmeans.fit_predict(data)

# Get centroids
centroids = kmeans.cluster_centers_

# Get inertia
inertia = kmeans.inertia_
```

## When to Use K-Means

✓ **Good for:**
- Spherical, well-separated clusters
- Large datasets (computationally efficient)
- Quick exploratory analysis

✗ **Not ideal for:**
- Non-spherical cluster shapes
- Clusters of vastly different sizes
- When k is unknown and hard to estimate

## Example: Income Clustering

In this project, we clustered people into 3 income groups based on Age and Income:
- Loaded income.csv dataset
- Applied K-Means with k=3
- Scaled features using MinMaxScaler to normalize Age and Income ranges
- Visualized resulting clusters with different colors
- Identified 3 distinct income segments

## Advantages & Disadvantages

| Pros | Cons |
|------|------|
| Fast and scalable | Need to specify k beforehand |
| Easy to implement | Sensitive to initial centroids |
| Works well with large datasets | Assumes spherical clusters |
| Interpretable results | Can get stuck in local optima |

## Related Concepts

- **K-Medoids**: Similar to K-Means but uses actual data points as centers
- **Hierarchical Clustering**: Alternative approach creating cluster dendrograms
- **DBSCAN**: Density-based clustering, good for arbitrary shapes 