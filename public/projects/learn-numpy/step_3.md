# Step 3: Array Operations and Broadcasting

Master the operations that make NumPy essential for AI: **vectorized operations** and **broadcasting**.

## What You'll Learn

- Element-wise operations (faster than loops!)
- Broadcasting rules
- Universal functions (ufuncs)
- Aggregation functions

---

## Element-Wise Operations

### The NumPy Magic: No Loops Needed!

```python
# Python way (slow) 😴
result = []
for i in range(len(arr1)):
    result.append(arr1[i] + arr2[i])

# NumPy way (fast) 🚀
result = arr1 + arr2  # Done!
```

### All Basic Operations Work

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b    # [11, 22, 33, 44]
a - b    # [-9, -18, -27, -36]
a * b    # [10, 40, 90, 160]
a / b    # [0.1, 0.1, 0.1, 0.1]
a ** 2   # [1, 4, 9, 16]
```

---

## Broadcasting: NumPy's Superpower

Broadcasting allows operations between arrays of **different shapes**!

### Example 1: Array + Scalar

```python
arr = np.array([1, 2, 3, 4])
result = arr + 10  # [11, 12, 13, 14]

# What happens: 10 is "broadcast" to [10, 10, 10, 10]
```

### Example 2: 2D + 1D

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

row = np.array([10, 20, 30])

result = matrix + row
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]

# The row is broadcast to each row of the matrix!
```

### Broadcasting Rules

Arrays can be broadcast together if:
1. They have the same shape, **OR**
2. One dimension is 1, **OR**
3. One dimension doesn't exist

```python
# Compatible shapes:
(3, 4) + (3, 4)   # ✅ Same shape
(3, 4) + (1, 4)   # ✅ Broadcasts first dimension
(3, 4) + (4,)     # ✅ Broadcasts to each row
(3, 1) + (1, 4)   # ✅ Broadcasts both

# Incompatible:
(3, 4) + (3,)     # ❌ Dimensions don't align
```

---

## Your Assignment

### Task 1: Normalize an Array

Create an array of 10 random numbers, then normalize it to range [0, 1].

```python
# Create random array
arr = np.random.rand(10) * 100  # Random 0-100

# Normalize: (x - min) / (max - min)
normalized = (arr - arr.min()) / (arr.max() - arr.min())

# Result should have min=0, max=1
```

**Expected output:** Array with values between 0 and 1

---

### Task 2: Matrix-Scalar Operations

Create a 3×3 matrix and perform these operations:
- Add 5 to all elements
- Multiply all elements by 2
- Square all elements

```python
matrix = np.arange(9).reshape(3, 3)

# Your operations here
```

**Expected output:** Three modified matrices

---

### Task 3: Broadcasting Practice

Create:
- A 4×3 matrix
- A 1D array of length 3

Add them together using broadcasting.

```python
matrix = np.ones((4, 3))
row = np.array([1, 2, 3])

# Add them - the row broadcasts to all 4 rows
```

**Expected output:** 4×3 matrix where each row has the added values

---

## Universal Functions (ufuncs)

NumPy provides optimized functions that work element-wise:

### Mathematical Functions

```python
arr = np.array([0, np.pi/2, np.pi])

np.sin(arr)    # Sine of each element
np.cos(arr)    # Cosine
np.exp(arr)    # e^x
np.log(arr)    # Natural log
np.sqrt(arr)   # Square root
```

### Rounding

```python
arr = np.array([1.2, 2.7, 3.5])

np.round(arr)   # [1., 3., 4.]
np.floor(arr)   # [1., 2., 3.]
np.ceil(arr)    # [2., 3., 4.]
```

---

## Aggregation Functions

Reduce arrays to single values or along axes:

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Global aggregations
arr.sum()        # 21 (all elements)
arr.mean()       # 3.5
arr.std()        # Standard deviation
arr.min()        # 1
arr.max()        # 6

# Along axes
arr.sum(axis=0)  # [5, 7, 9] - sum each column
arr.sum(axis=1)  # [6, 15] - sum each row
```

**Axis 0** = down columns  
**Axis 1** = across rows

---

## AI/ML Application: Data Normalization

In machine learning, we often normalize data:

```python
# Example: Normalize features for neural network
data = np.random.randn(1000, 10)  # 1000 samples, 10 features

# Z-score normalization (mean=0, std=1)
mean = data.mean(axis=0)
std = data.std(axis=0)
normalized_data = (data - mean) / std

# Min-max normalization (range 0-1)
min_val = data.min(axis=0)
max_val = data.max(axis=0)
normalized_data = (data - min_val) / (max_val - min_val)
```

This is **exactly** what happens in AI preprocessing!

---

## Common Mistakes to Avoid

❌ **Forgetting about broadcasting**
```python
arr = np.array([1, 2, 3])
arr + [10, 20]  # Error! Shapes don't match
```

❌ **Wrong axis**
```python
matrix.sum(axis=1)  # Sums rows (result: 1D)
matrix.sum(axis=0)  # Sums columns (result: 1D)
```

❌ **Modifying original array**
```python
arr = np.array([1, 2, 3])
arr2 = arr          # ❌ Same object!
arr2 = arr.copy()   # ✅ New copy
```

---

## Complete the Assignment

Open the Colab notebook and complete the normalization and broadcasting tasks.

**We check your results, not your variable names!** Name things however you like. 😊

---

## Next: Fancy Indexing

Step 4 will cover:
- Boolean indexing (filter arrays)
- Fancy indexing (use arrays as indices)
- Where and masking
