# Pandas DataFrame - Complete Guide

## Overview
A **DataFrame** is a 2D labeled data structure in Pandas with rows and columns. It's like a spreadsheet or database table - perfect for data analysis and manipulation.

## Creating DataFrames

### Method 1: From CSV File
```python
import pandas as pd

# Read CSV file
df = pd.read_csv("data.csv")
```

### Method 2: From Dictionary
```python
weather_data = {
    'day': ['1/1/2017','1/2/2017','1/3/2017'],
    'temperature': [32, 35, 28],
    'windspeed': [6, 7, 2],
    'event': ['Rain', 'Sunny', 'Snow']
}
df = pd.DataFrame(weather_data)
```

### Method 3: From Lists or Arrays
```python
df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
```

## DataFrame Properties

### Shape
```python
df.shape          # Returns (rows, columns) - e.g., (6, 4)
df.columns        # Returns all column names
df.index          # Returns row indices
df.dtypes         # Returns data type of each column
```

### View Data
```python
df                # Display entire DataFrame
df.head()         # First 5 rows
df.tail()         # Last 5 rows
df.info()         # Summary of DataFrame
df.describe()     # Statistical summary (count, mean, std, min, max, etc.)
```

## Accessing Data

### Single Column
```python
# Both methods are equivalent
df.day                    # Using dot notation
df["day"]                 # Using bracket notation

# Returns a Series (1D array)
type(df["event"])        # <class 'pandas.core.series.Series'>
```

### Multiple Columns
```python
df[["event", "day"]]     # Returns a DataFrame with 2 columns
df[["day", "temperature", "windspeed"]]
```

### Row Slicing
```python
df[2:5]                  # Rows 2 to 4 (excludes 5) - uses iloc internally
```

## Filtering Data

### Boolean Filtering
```python
# Find rows where temperature >= 32
df[df.temperature >= 32]

# Multiple conditions
df[(df.temperature >= 32) & (df.windspeed > 5)]
df[(df.temperature >= 32) | (df.event == 'Snow')]
```

### Finding Specific Values
```python
# Find row(s) with maximum temperature
df[df.temperature == df.temperature.max()]

# Find specific values
df[df.event == 'Snow']
df[df.day == '1/3/2017']
```

## Index Management

### Understanding Index
```python
df.index              # Returns the index labels
type(df.index)        # <class 'pandas.core.indexes.base.Index'>
```

### Accessing with Index - loc
```python
# Access row by index label
df.loc["1/3/2017"]    # If date is the index

# Access specific cell
df.loc["1/3/2017", "temperature"]

# Access multiple rows
df.loc[["1/1/2017", "1/2/2017"]]
```

### Reset Index
```python
df.reset_index(inplace=True)   # Convert index to regular column
df                             # Index becomes 0, 1, 2, ...
```

### Set Custom Index
```python
df.set_index("event", inplace=True)    # Set 'event' column as index
df.loc["Snow"]                         # Access all rows with 'Snow' event
```

## Statistical Operations

### Summary Statistics
```python
df.describe()              # Count, mean, std, min, 25%, 50%, 75%, max

# Specific statistics
df.mean()                  # Mean of each numeric column
df.sum()                   # Sum of each column
df.min()                   # Minimum of each column
df.max()                   # Maximum of each column
df.std()                   # Standard deviation
df.median()                # Median
df.var()                   # Variance

# For specific column
df["temperature"].mean()
df["temperature"].max()
```

## Data Manipulation

### Adding Columns
```python
df["new_column"] = [1, 2, 3, 4, 5, 6]      # Add new column
df["celsius"] = (df["temperature"] - 32) * 5/9  # Calculate from existing
```

### Removing Columns
```python
df.drop("temperature", axis=1, inplace=True)   # Remove column
df = df.drop("new_column", axis=1)             # Or assign back
```

### Renaming Columns
```python
df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.columns = ['day', 'temp', 'wind', 'weather']  # Rename all
```

### Handling Missing Values
```python
df.fillna(0, inplace=True)                # Fill NaN with 0
df.fillna({'col1': 0, 'col2': 'Unknown'})  # Fill with different values
df.dropna()                               # Remove rows with NaN
df.isnull()                               # Check for NaN values
```

## Sorting and Grouping

### Sorting
```python
df.sort_values("temperature")             # Sort by column
df.sort_values("temperature", ascending=False)  # Descending order
df.sort_values(["temperature", "day"])    # Sort by multiple columns
```

### Grouping
```python
df.groupby("event").mean()                # Mean for each event type
df.groupby("event").count()               # Count by event
df.groupby("event")["temperature"].max()  # Max temperature by event
```

## Data Types

### Check Data Types
```python
df.dtypes                                 # Data type of each column
df["temperature"].dtype                   # Data type of specific column
```

### Convert Data Types
```python
df["temperature"] = df["temperature"].astype(int)    # To integer
df["temperature"] = df["temperature"].astype(float)  # To float
df["day"] = pd.to_datetime(df["day"])                # To datetime
```

## Common Operations

```python
# Count rows
len(df)
df.shape[0]

# Count specific values
df["event"].value_counts()    # Count each event type

# Unique values
df["event"].unique()          # Get unique events
df["event"].nunique()         # Number of unique events

# Get row by position (iloc)
df.iloc[0]                    # First row
df.iloc[0:3]                  # First 3 rows
df.iloc[0, 1]                 # Row 0, Column 1
```

## DataFrame vs Series

| Feature | DataFrame | Series |
|---------|-----------|--------|
| Dimensions | 2D (rows × columns) | 1D (single column) |
| Created from | Dictionary, CSV, lists | Single column, list |
| Access | `df["col"]` returns Series | Direct values |
| Operations | Operations on multiple columns | Operations on single column |

Example:
```python
df["temperature"]             # Returns a Series
df[["temperature"]]           # Returns a DataFrame
```

## Exporting Data

```python
# Save to CSV
df.to_csv("output.csv", index=False)

# Save to Excel
df.to_excel("output.xlsx", index=False)

# Save to JSON
df.to_json("output.json")

# Convert to dictionary
df.to_dict()
```

## Important Tips

1. **Column Access**: Use `df["column"]` or `df.column` (both work, but bracket notation is safer)
2. **Index**: Set meaningful index for easier data retrieval with `loc`
3. **Boolean Indexing**: Use `&` (and) and `|` (or), NOT `and`/`or`
4. **Chaining**: You can chain operations: `df[df.temp > 30].sort_values("day")`
5. **In-place operations**: Use `inplace=True` to modify the original DataFrame
6. **Copy data**: Use `df.copy()` to avoid unintended modifications

## Quick Commands Reference

```python
# Check data
df.head(), df.tail(), df.info(), df.describe()

# Access data
df["col"], df[["col1", "col2"]], df[2:5], df.loc[], df.iloc[]

# Filter
df[df.col > value], df[df.col == "text"]

# Modify
df["new_col"] = ..., df.drop(), df.rename()

# Statistics
df.mean(), df.sum(), df.max(), df.min(), df.describe()

# Data type
df.dtypes, df.astype()

# Index
df.reset_index(), df.set_index(), df.index

# Export
df.to_csv(), df.to_excel(), df.to_json()
```

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| KeyError | Column doesn't exist | Check column name spelling |
| TypeError: invalid value | Wrong type in fillna | Use appropriate type for column |
| NameError | Missing import | `import pandas as pd` |
| AttributeError | Wrong method name | Check Pandas documentation |

## Practice Dataset Structure
```
day         temperature  windspeed  event
1/1/2017    32          6          Rain
1/2/2017    35          7          Sunny
1/3/2017    28          2          Snow
1/4/2017    24          7          Snow
1/5/2017    32          4          Rain
1/6/2017    31          2          Sunny
```

## Summary

DataFrames are essential for:
- Loading and exploring data from CSV/databases
- Data cleaning and preprocessing
- Statistical analysis
- Data visualization preparation
- Machine learning data preparation

Master DataFrames to excel in data science and analytics!
