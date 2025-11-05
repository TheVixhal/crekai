# Step 2: Data Selection and Filtering

Master the art of selecting exactly the data you need from DataFrames!

## What You'll Learn

- loc and iloc indexing
- Boolean filtering
- Query operations
- Selecting columns and rows

---

## Your Assignment

### Task 1: Using loc and iloc

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['NY', 'SF', 'LA', 'Boston', 'Seattle'],
    'Salary': [50000, 60000, 55000, 52000, 58000]
})

# loc: label-based indexing
row = df.loc[0]                    # First row
subset = df.loc[0:2, 'Name':'Age'] # Rows 0-2, cols Name to Age

# iloc: position-based indexing
row = df.iloc[0]                   # First row
subset = df.iloc[0:3, 0:2]         # First 3 rows, first 2 cols

# Select single value
value = df.loc[2, 'Salary']        # Charlie's salary
value = df.iloc[2, 3]              # Same thing
```

**Expected output:** Various DataFrame subsets

---

### Task 2: Boolean Filtering

```python
# Filter rows where Age > 28
filtered = df[df['Age'] > 28]

# Multiple conditions (AND)
filtered = df[(df['Age'] > 25) & (df['Salary'] > 52000)]

# Multiple conditions (OR)
filtered = df[(df['City'] == 'NY') | (df['City'] == 'SF')]

# NOT condition
filtered = df[~(df['Age'] < 30)]  # Age NOT less than 30

# Using isin()
cities = ['NY', 'SF', 'LA']
filtered = df[df['City'].isin(cities)]
```

**Expected output:** Filtered DataFrames

---

### Task 3: Query Method

```python
# Query with string expression
filtered = df.query('Age > 28 and Salary > 52000')

# Using variables
min_age = 30
filtered = df.query('Age > @min_age')

# String matching
filtered = df.query('City == "NY" or City == "SF"')
```

**Expected output:** Query results

---

## Advanced Selection

### Selecting Columns

```python
# Single column (returns Series)
ages = df['Age']

# Multiple columns (returns DataFrame)
subset = df[['Name', 'Salary']]

# Columns by position
subset = df.iloc[:, 0:2]  # First 2 columns

# All except some columns
subset = df.drop(['Age'], axis=1)
```

### Conditional Column Selection

```python
# Numeric columns only
numeric_df = df.select_dtypes(include=['number'])

# String columns only
string_df = df.select_dtypes(include=['object'])

# Columns containing 'a'
cols_with_a = df.filter(like='a', axis=1)
```

---

## Modifying Data

### Update values

```python
# Update single value
df.loc[0, 'Age'] = 26

# Update entire column
df['Age'] = df['Age'] + 1

# Conditional update
df.loc[df['Age'] > 30, 'Age'] = 30  # Cap at 30
```

### Add new columns

```python
# From calculation
df['Tax'] = df['Salary'] * 0.2

# From condition
df['Senior'] = df['Age'] > 30

# From apply function
df['Name_Length'] = df['Name'].apply(len)
```

---

## String Operations

```python
# String methods
df['Name'].str.upper()           # ALICE, BOB, ...
df['Name'].str.lower()           # alice, bob, ...
df['Name'].str.len()             # 5, 3, 7, ...
df['Name'].str.contains('a')     # True, False, ...
df['Name'].str.startswith('A')   # True, False, ...
```

---

## Complete the Assignment

Practice selection and filtering in Colab:
1. Use loc and iloc effectively
2. Filter data with boolean conditions
3. Use query for complex filters

---

## Next: Data Cleaning

Step 3 covers handling missing data and duplicates.
