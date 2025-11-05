# Step 4: Boolean Indexing and Filtering

Learn how to **filter and manipulate data** using conditions - a crucial skill for data science and AI!

## What You'll Learn

- Boolean indexing
- Conditional operations
- np.where() function
- Masking arrays

---

## Boolean Indexing

### Filter Arrays with Conditions

```python
arr = np.array([1, 5, 3, 8, 2, 9, 4])

# Create a boolean mask
mask = arr > 5
# [False, False, False, True, False, True, False]

# Use mask to filter
result = arr[mask]  # [8, 9]

# Or in one line:
result = arr[arr > 5]  # [8, 9]
```

**This is how we filter data in AI preprocessing!**

---

## Your Assignment

### Task 1: Filter Positive Numbers

Create an array with both positive and negative numbers. Extract only the positive ones.

```python
arr = np.array([-5, 3, -2, 8, -1, 6, 0, -3, 4])

# Filter for positive numbers (> 0)
positive = arr[arr > 0]
```

**Expected output:** Array containing only positive values

---

### Task 2: Replace Values with np.where()

Create an array of 20 random numbers (0-100). Replace all values:
- Less than 30 → set to 0
- Greater than 70 → set to 100
- Keep others unchanged

```python
arr = np.random.randint(0, 101, 20)

# Method 1: np.where
arr = np.where(arr < 30, 0, arr)
arr = np.where(arr > 70, 100, arr)

# Method 2: Boolean indexing
arr[arr < 30] = 0
arr[arr > 70] = 100
```

**Expected output:** Array with transformed values

---

### Task 3: Count Elements Meeting Condition

Create an array of 50 random numbers (0-10). Count how many are:
- Greater than 5
- Equal to 0
- Between 3 and 7 (inclusive)

```python
arr = np.random.randint(0, 11, 50)

# Count using sum() on boolean array
count_gt_5 = np.sum(arr > 5)
count_zero = np.sum(arr == 0)
count_between = np.sum((arr >= 3) & (arr <= 7))
```

**Expected output:** Three count values

---

## Advanced Boolean Operations

### Combine Conditions

```python
arr = np.array([1, 5, 3, 8, 2, 9, 4])

# AND: &
arr[(arr > 2) & (arr < 8)]  # [5, 3, 4]

# OR: |
arr[(arr < 3) | (arr > 7)]  # [1, 2, 8, 9]

# NOT: ~
arr[~(arr > 5)]  # [1, 5, 3, 2, 4]
```

**Important:** Use `&` and `|`, not `and` and `or`!

---

## The np.where() Function

Three powerful uses:

### 1. If-Else Replacement

```python
arr = np.array([1, 2, 3, 4, 5])

# If > 3, set to 1, else set to 0
result = np.where(arr > 3, 1, 0)  # [0, 0, 0, 1, 1]
```

### 2. Find Indices

```python
arr = np.array([1, 5, 3, 8, 2])

# Get indices where condition is True
indices = np.where(arr > 4)  # (array([1, 3]),)
```

### 3. Complex Replacements

```python
arr = np.random.randint(0, 100, 20)

# Categorize: low (<30), medium (30-70), high (>70)
categories = np.where(arr < 30, 'low',
                np.where(arr > 70, 'high', 'medium'))
```

---

## Real AI/ML Example

### Data Cleaning: Handle Outliers

```python
# Dataset with outliers
data = np.random.randn(1000) * 10

# Clip outliers to 3 standard deviations
mean = data.mean()
std = data.std()
threshold = 3 * std

# Replace outliers with threshold
data = np.where(data > mean + threshold, mean + threshold, data)
data = np.where(data < mean - threshold, mean - threshold, data)
```

### Missing Data Handling

```python
# Replace negative values (missing data) with mean
data = np.array([1.5, -1, 3.2, -1, 2.8, 4.1])

mask = data > 0
mean_valid = data[mask].mean()

data = np.where(data < 0, mean_valid, data)
```

---

## Masking vs Direct Indexing

### Masking (Returns filtered array)
```python
arr = np.array([1, 2, 3, 4, 5])
filtered = arr[arr > 2]  # [3, 4, 5] - new array
```

### Direct Modification
```python
arr = np.array([1, 2, 3, 4, 5])
arr[arr > 2] = 0  # [1, 2, 0, 0, 0] - modifies original
```

---

## Performance Tips

```python
# Slow: Python loop
result = []
for x in arr:
    if x > 5:
        result.append(x * 2)

# Fast: NumPy vectorized
result = arr[arr > 5] * 2

# Even faster: In-place operations
arr[arr > 5] *= 2
```

**100x faster for large arrays!** ⚡

---

## Complete the Assignment

Open the Colab notebook and implement:
1. Filtering positive numbers
2. Conditional replacements
3. Counting with conditions

The system validates your **logic**, not variable names!

---

## Next: Linear Algebra

In Step 5, you'll learn:
- Matrix multiplication
- Dot products
- Solving linear systems
- Eigenvalues and eigenvectors
