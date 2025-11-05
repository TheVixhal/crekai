# Step 5: Linear Algebra with NumPy

Master the mathematical foundation of AI: **linear algebra**. Neural networks are built on these operations!

## What You'll Learn

- Matrix multiplication
- Dot products
- Transpose operations
- Linear algebra functions

---

## Why Linear Algebra Matters in AI

### Neural Network = Matrix Multiplications!

```python
# Input: 1 sample, 784 features (28×28 image flattened)
X = np.random.randn(1, 784)

# Weights: 784 inputs → 128 neurons
W1 = np.random.randn(784, 128)

# Forward pass: Matrix multiplication!
hidden = X @ W1  # Shape: (1, 128)

# Next layer: 128 → 10 outputs
W2 = np.random.randn(128, 10)
output = hidden @ W2  # Shape: (1, 10)
```

**This IS a neural network!** (Simplified, but the math is the same)

---

## Matrix Multiplication

### The @ Operator (Recommended)

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = A @ B  # Matrix multiplication
# [[19, 22],
#  [43, 50]]
```

### Rules for Matrix Multiplication

```python
# For A @ B to work:
# - A.shape = (m, n)
# - B.shape = (n, p)
# - Result = (m, p)

(3, 4) @ (4, 2)  # ✅ Result: (3, 2)
(2, 5) @ (5, 3)  # ✅ Result: (2, 3)
(3, 4) @ (3, 2)  # ❌ Error! Inner dimensions don't match
```

**Remember:** Columns of A must equal rows of B!

---

## Your Assignment

### Task 1: Matrix Multiplication

Create two matrices and multiply them:
- Matrix A: 3×4
- Matrix B: 4×2
- Result C: 3×2

```python
A = np.random.randint(1, 10, (3, 4))
B = np.random.randint(1, 10, (4, 2))

C = A @ B  # or np.matmul(A, B) or A.dot(B)
```

**Expected output:** 3×2 matrix

---

### Task 2: Transpose

Create a matrix and compute its transpose. Verify that `(A @ B).T = B.T @ A.T`.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Transpose
A_T = A.T  # or np.transpose(A)

# Shape changes: (2, 3) → (3, 2)
```

**Expected output:** Transposed matrix with swapped dimensions

---

### Task 3: Dot Product

Calculate the dot product of two vectors and verify it equals the sum of element-wise products.

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Verify
manual = np.sum(v1 * v2)  # Should equal dot
```

**Expected output:** Single number (the dot product)

---

## Essential Linear Algebra Functions

### Matrix Operations

```python
A = np.array([[1, 2],
              [3, 4]])

# Determinant
np.linalg.det(A)  # -2.0

# Inverse
A_inv = np.linalg.inv(A)
A @ A_inv  # Identity matrix

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### Solving Linear Systems

```python
# Solve: Ax = b
A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)  # [2, 3]

# Verify
np.allclose(A @ x, b)  # True
```

---

## AI/ML Applications

### 1. Neural Network Forward Pass

```python
# Batch of 32 images, 784 pixels each
X = np.random.randn(32, 784)

# Weights for first layer
W = np.random.randn(784, 128)
b = np.random.randn(128)

# Forward pass
output = X @ W + b  # Broadcasting adds bias!
# Shape: (32, 128)
```

### 2. Cosine Similarity (Text/Image Similarity)

```python
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2)

# Compare two embeddings
embedding1 = np.random.randn(512)
embedding2 = np.random.randn(512)

similarity = cosine_similarity(embedding1, embedding2)
# Range: -1 (opposite) to 1 (identical)
```

### 3. PCA (Principal Component Analysis)

```python
# Dimensionality reduction
data = np.random.randn(1000, 50)  # 1000 samples, 50 features

# Center the data
data_centered = data - data.mean(axis=0)

# Compute covariance matrix
cov = np.cov(data_centered.T)

# Get eigenvalues/eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Project onto top 10 components
top_10 = eigenvectors[:, :10]
data_reduced = data_centered @ top_10  # (1000, 10)
```

---

## Key Takeaways

🎯 **@ operator** - Matrix multiplication (use this!)  
🎯 **Broadcasting** - Adds flexibility (bias terms, etc.)  
🎯 **Transpose** - Changes row/column orientation  
🎯 **np.linalg** - Your linear algebra toolbox  

---

## Performance Comparison

```python
# Python loops (slow)
def matrix_mult_python(A, B):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

# NumPy (fast)
C = A @ B

# Speed: NumPy is 100-1000x faster! 🚀
```

---

## Complete the Assignment

Practice matrix operations in the Colab notebook:
1. Matrix multiplication with proper dimensions
2. Transpose verification
3. Dot product calculation

---

## Congratulations! 🎉

You've completed the NumPy fundamentals! You now know:
- ✅ Array creation and manipulation
- ✅ Indexing and slicing
- ✅ Broadcasting and operations
- ✅ Boolean indexing
- ✅ Linear algebra

**You're ready for:** PyTorch, TensorFlow, and building real AI models!

---

## What's Next?

Continue your AI journey with:
- **Learn PyTorch** - Deep learning framework
- **Learn Pandas** - Data manipulation
- **Build ANN** - Create your first neural network
- **Build GPT** - Build a transformer from scratch

Keep learning! 🚀
