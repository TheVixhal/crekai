# Step 2: Multi-Dimensional Arrays and Indexing

Now that you understand 1D arrays, let's explore NumPy's true power: **multi-dimensional arrays** (matrices and tensors).

## What You'll Learn

- Creating 2D arrays (matrices)
- Indexing and slicing arrays
- Reshaping arrays
- Array properties and attributes

---

## Understanding Dimensions

### 1D Array (Vector)
```python
arr_1d = np.array([1, 2, 3, 4, 5])
# Shape: (5,)
```

### 2D Array (Matrix)
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
# Shape: (2, 3) - 2 rows, 3 columns
```

### 3D Array (Tensor)
```python
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])
# Shape: (2, 2, 2)
```

**In AI/ML:** Images are 3D arrays (height × width × channels)!

---

## Your Assignment

### Task 1: Create a 3×3 Matrix

Create a 3×3 matrix with any numbers you want.

```python
# Method 1: From list of lists
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Method 2: Using arange and reshape
matrix = np.arange(1, 10).reshape(3, 3)

# Method 3: Special matrices
matrix = np.eye(3)        # Identity matrix
matrix = np.zeros((3,3))  # All zeros
matrix = np.random.rand(3, 3)  # Random numbers
```

**Expected output:** A 3×3 NumPy array

---

### Task 2: Index and Slice

From your 3×3 matrix, extract:
- The center element (middle row, middle column)
- The first row
- The last column

```python
# Indexing starts at 0!
center = matrix[1, 1]      # Row 1, Col 1
first_row = matrix[0]      # or matrix[0, :]
last_col = matrix[:, 2]    # All rows, last column
```

**Expected output:** Three separate values/arrays

---

### Task 3: Reshape an Array

Create a 1D array of 12 elements, then reshape it to:
- 3×4 matrix
- 2×6 matrix
- 4×3 matrix

```python
# Start with 1D array
arr = np.arange(12)

# Reshape to different dimensions
reshaped_3x4 = arr.reshape(3, 4)
reshaped_2x6 = arr.reshape(2, 6)
# Your turn for 4x3!
```

**Expected output:** Three different shaped arrays

---

## Key Concepts

### Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape      # (2, 3) - dimensions
arr.ndim       # 2 - number of dimensions
arr.size       # 6 - total elements
arr.dtype      # int64 - data type
```

### Indexing

```python
arr[0, 1]      # Element at row 0, col 1 → 2
arr[1]         # Entire row 1 → [4, 5, 6]
arr[:, 2]      # Entire column 2 → [3, 6]
arr[0:2, 1:3]  # Slice: rows 0-1, cols 1-2
```

### Reshaping Rules

```python
# Total elements must match!
arr = np.arange(12)  # 12 elements

arr.reshape(3, 4)    # ✅ 3*4 = 12
arr.reshape(2, 6)    # ✅ 2*6 = 12
arr.reshape(3, 5)    # ❌ 3*5 = 15 (Error!)
```

---

## Why This Matters for AI

### Images as Arrays
```python
# RGB image: 224×224 pixels, 3 color channels
image = np.zeros((224, 224, 3))

# Grayscale image
gray_image = np.zeros((224, 224))
```

### Neural Network Weights
```python
# Layer 1: 784 inputs → 128 neurons
weights_1 = np.random.randn(784, 128)

# Layer 2: 128 → 10 outputs
weights_2 = np.random.randn(128, 10)
```

### Batches of Data
```python
# 32 images, each 28×28 pixels
batch = np.zeros((32, 28, 28))
```

---

## Common Patterns

### Creating Special Matrices

```python
np.eye(3)          # Identity matrix (1s on diagonal)
np.diag([1, 2, 3]) # Diagonal matrix
np.ones((3, 3))    # All ones
np.full((3, 3), 7) # All sevens
```

### Combining Arrays

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.vstack([a, b])  # Stack vertically → 2×3 matrix
np.hstack([a, b])  # Stack horizontally → [1,2,3,4,5,6]
```

---

## Practice Tips

🎯 **Visualize shapes** - Draw arrays on paper to understand dimensions  
🎯 **Use .shape constantly** - Always check what shape you have  
🎯 **Experiment with reshape** - Try different combinations  
🎯 **Remember indexing** - [row, column] for 2D arrays  

---

## Complete the Assignment

Open the Colab notebook and complete all tasks. The verification will check:
- ✅ You created a 3×3 matrix
- ✅ You extracted the correct elements
- ✅ You reshaped arrays correctly

**No specific variable names required** - we check your OUTPUT is correct!

---

## Next Up: Array Operations

In Step 3, you'll learn:
- Element-wise operations
- Broadcasting
- Aggregations (sum, mean, etc.)
