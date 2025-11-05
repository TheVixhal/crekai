# Step 3: Data Cleaning

Learn to clean messy real-world data - the most important skill in data science!

## Your Assignment

### Task 1: Handle Missing Data

```python
import pandas as pd
import numpy as np

# DataFrame with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, np.nan]
})

# Detect missing values
print(df.isnull())
print(df.isnull().sum())

# Drop rows with any missing values
clean_df = df.dropna()

# Drop columns with any missing values
clean_df = df.dropna(axis=1)

# Fill missing values
filled_df = df.fillna(0)  # Fill with 0
filled_df = df.fillna(df.mean())  # Fill with mean
filled_df = df.fillna(method='ffill')  # Forward fill
```

**Expected output:** Clean DataFrame without NaN values

---

### Task 2: Remove Duplicates

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Score': [85, 90, 85, 95, 90]
})

# Find duplicates
print(df.duplicated())

# Remove duplicates
clean_df = df.drop_duplicates()

# Remove based on specific column
clean_df = df.drop_duplicates(subset=['Name'], keep='first')
```

**Expected output:** DataFrame without duplicates

---

### Task 3: Data Type Conversion

```python
df = pd.DataFrame({
    'Age': ['25', '30', '35'],
    'Price': ['100.50', '200.75', '150.25']
})

# Convert to numeric
df['Age'] = df['Age'].astype(int)
df['Price'] = df['Price'].astype(float)

# Handle errors
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
```

**Expected output:** Correctly typed columns
