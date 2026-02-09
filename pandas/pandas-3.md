# Pandas: Handling Missing Data & Time Series Indexing

This document covers techniques for handling missing data (`NaN`) and setting up time-series indices in Pandas, corresponding to the topics discussed.

## 1. Convert String Column to Date Type
When importing data, date columns are often read as strings (object type). It is crucial to convert them to datetime objects for time-series analysis.

```python
import pandas as pd

df = pd.read_csv("weather_data.csv")

# Convert 'day' column to datetime
df['day'] = pd.to_datetime(df['day'])
```

## 2. Set Date as Index
Setting the date column as the DataFrame index allows for easy slicing and plotting by date.

```python
# Set 'day' as the index. inplace=True modifies the original dataframe.
df.set_index('day', inplace=True)
```

## 3. Handling Missing Data: `fillna()`
The `fillna()` method fills missing values (`NaN`) with specified values.

### 3.1 Basic Usage (Fill with a single value)
```python
# Replace all NaN with 0
new_df = df.fillna(0)

# Replace with specific values per column
new_df = df.fillna({
    'temperature': 0,
    'windspeed': 0,
    'event': 'No Event'
})
```

### 3.2 Forward Fill (`method="ffill"`)
Propagate the last valid observation forward to the next valid one. Useful for data where the value doesn't change frequently.
```python
new_df = df.fillna(method="ffill")
```

### 3.3 Backward Fill (`method="bfill"`)
Use the next valid observation to fill the gap backwards.
```python
new_df = df.fillna(method="bfill")
```

### 3.4 Axis Parameter
Determines the direction of filling.
*   `axis="index"` or `0`: Fills vertically (default).
*   `axis="columns"` or `1`: Fills horizontally (copies value from the previous column).
```python
new_df = df.fillna(method="bfill", axis="columns")
```

### 3.5 Limit Parameter
Limits the number of consecutive `NaN` values to fill.
```python
# Only fill the first 1 missing value in a consecutive series of NaNs
new_df = df.fillna(method="ffill", limit=1)
```

## 4. Interpolation: `interpolate()`
Interpolation uses valid values to estimate missing ones, providing a smoother curve than simple filling.

### 4.1 Linear Interpolation (Default)
Tries to guess the missing value by drawing a straight line between the previous and next points.
```python
new_df = df.interpolate()
```

### 4.2 Time Interpolation (`method="time"`)
If your index is a DatetimeIndex, `method="time"` considers the actual time intervals. This is critical if your data points are not equally spaced in time.
```python
new_df = df.interpolate(method="time")
```

## 5. Dropping Missing Data: `dropna()`
Sometimes it is better to drop missing data completely rather than filling it.

### 5.1 Basic Usage (Drop rows with NaNs)
Drops any row that has at least one missing value.
```python
new_df = df.dropna()
```

### 5.2 How Parameter
*   `how="any"`: Default. Drops row if *any* value is `NaN`.
*   `how="all"`: Drops row only if *all* values in that row are `NaN`.
```python
new_df = df.dropna(how="all")
```

### 5.3 Thresh Parameter
Requires a minimum number of non-NA values to keep the row.
```python
# Keep row only if it has at least 1 valid (non-NaN) value
new_df = df.dropna(thresh=1)
```
