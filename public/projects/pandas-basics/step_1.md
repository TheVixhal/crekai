# Step 1: Introduction to Pandas

Master Pandas - the essential library for data manipulation and analysis in Python!

## What is Pandas?

Pandas provides powerful data structures for working with structured data:
- **DataFrames** - 2D tables (like Excel/SQL)
- **Series** - 1D labeled arrays
- **Data I/O** - Read CSV, Excel, JSON, SQL
- **Data cleaning** - Handle missing data, duplicates

## Why Pandas for AI/ML?

🎯 **Every AI project starts with data**  
📊 **80% of ML is data preparation**  
🔧 **Pandas makes it easy**  

Used by: Data scientists, ML engineers, analysts worldwide

---

## Your Assignment

### Task 1: Create a DataFrame

Create a DataFrame from a dictionary:

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['NY', 'SF', 'LA', 'Boston'],
    'Salary': [50000, 60000, 55000, 52000]
}

df = pd.DataFrame(data)
print(df)
```

**Expected output:** DataFrame with 4 rows, 4 columns

---

### Task 2: Basic DataFrame Operations

```python
# View first rows
print(df.head())

# Info about DataFrame
print(df.info())

# Statistical summary
print(df.describe())

# Select column
ages = df['Age']  # Returns Series

# Select multiple columns
subset = df[['Name', 'Salary']]
```

**Expected output:** Various DataFrame views

---

### Task 3: Create a Series

```python
# From list
s = pd.Series([10, 20, 30, 40, 50])

# With custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# From dict
s = pd.Series({'a': 10, 'b': 20, 'c': 30})

print(s)
```

**Expected output:** Series with values and index

---

## DataFrames: The Basics

### Creating DataFrames

```python
# Method 1: From dict
df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6]
})

# Method 2: From lists
df = pd.DataFrame(
    [[1, 4], [2, 5], [3, 6]],
    columns=['col1', 'col2']
)

# Method 3: From NumPy
import numpy as np
df = pd.DataFrame(np.random.rand(3, 3), 
                  columns=['A', 'B', 'C'])
```

### Viewing Data

```python
df.head(10)     # First 10 rows
df.tail(5)      # Last 5 rows
df.sample(3)    # Random 3 rows
df.shape        # (rows, cols)
df.columns      # Column names
df.index        # Row indices
```

---

## Series: 1D Data

```python
s = pd.Series([1, 2, 3, 4, 5], 
              index=['a', 'b', 'c', 'd', 'e'])

s['a']          # Access by label → 1
s[0]            # Access by position → 1
s['a':'c']      # Slice by label
s[0:2]          # Slice by position

s.values        # Get NumPy array
s.index         # Get index
```

---

## Reading Data

### From CSV

```python
# Read CSV file
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv',
                 sep=',',
                 header=0,
                 index_col=0,
                 nrows=1000)
```

### From Other Sources

```python
# Excel
df = pd.read_excel('data.xlsx')

# JSON
df = pd.read_json('data.json')

# SQL
df = pd.read_sql(query, connection)

# Clipboard
df = pd.read_clipboard()  # Copy from Excel, paste to Pandas!
```

---

## Writing Data

```python
# To CSV
df.to_csv('output.csv', index=False)

# To Excel
df.to_excel('output.xlsx', index=False)

# To JSON
df.to_json('output.json')
```

---

## Quick Data Inspection

```python
df.info()         # Column types, non-null counts
df.describe()     # Statistics (mean, std, min, max)
df.dtypes         # Data type of each column
df.isnull().sum() # Count missing values per column
df.nunique()      # Count unique values per column
```

---

## Real Example: Load and Explore

```python
# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# Quick exploration
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\\nFirst 5 rows:")
print(df.head())
print(f"\\nSummary stats:")
print(df.describe())
print(f"\\nMissing values:")
print(df.isnull().sum())
```

---

## Complete the Assignment

Practice in Colab:
1. Create DataFrames from different sources
2. Perform basic operations
3. Create and manipulate Series

---

## Next: Data Selection and Filtering

Step 2 covers:
- loc and iloc
- Boolean indexing
- Query operations
- Column operations
