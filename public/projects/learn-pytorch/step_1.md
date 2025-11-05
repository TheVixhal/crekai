# Step 1: Introduction to PyTorch

Welcome to PyTorch! The most popular deep learning framework used by researchers and industry leaders like OpenAI, Meta, and Tesla.

## What is PyTorch?

PyTorch is a deep learning framework that provides:
- **Tensors** - Like NumPy arrays but GPU-accelerated
- **Autograd** - Automatic differentiation for backpropagation
- **Neural Network modules** - Pre-built layers and models
- **GPU support** - Train models 100x faster

## Why PyTorch?

🔥 **Used by:** OpenAI (ChatGPT), Meta (LLaMA), Tesla (Autopilot)  
🎓 **Easy to learn** - Pythonic and intuitive  
🚀 **Research favorite** - Dynamic computation graphs  
⚡ **Production ready** - TorchServe for deployment  

---

## Your First Assignment

### Task 1: Create Tensors

Create PyTorch tensors in different ways:

```python
import torch

# From Python list
tensor1 = torch.tensor([1, 2, 3, 4, 5])

# Using built-in functions
tensor2 = torch.zeros(3, 3)      # 3×3 zeros
tensor3 = torch.ones(2, 4)       # 2×4 ones
tensor4 = torch.rand(3, 3)       # 3×3 random (0-1)
tensor5 = torch.randn(3, 3)      # 3×3 normal distribution

# Like np.arange
tensor6 = torch.arange(0, 10)    # [0, 1, ..., 9]
```

**Expected output:** Various tensors with different shapes

---

### Task 2: Tensor to NumPy and Back

Convert between PyTorch tensors and NumPy arrays:

```python
import numpy as np

# NumPy to PyTorch
np_array = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(np_array)

# PyTorch to NumPy
tensor = torch.tensor([1.0, 2.0, 3.0])
np_array = tensor.numpy()
```

**Expected output:** Successful conversions in both directions

---

### Task 3: Tensor Operations

Perform basic operations on tensors:

```python
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([10, 20, 30, 40, 50])

# Element-wise operations
add = a + b
sub = a - b
mul = a * b
div = a / b

# Aggregations
mean = a.float().mean()
std = a.float().std()
sum_val = a.sum()
```

**Expected output:** Results from various operations

---

## Tensors vs NumPy Arrays

### Similarities
- Both are multi-dimensional arrays
- Similar syntax and operations
- Element-wise operations
- Broadcasting

### Differences

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| **Device** | CPU only | CPU or GPU |
| **Gradients** | No | Yes (autograd) |
| **Speed** | Fast | Faster (GPU) |
| **Use case** | Data processing | Deep learning |

---

## Tensor Properties

```python
tensor = torch.randn(3, 4, 5)

tensor.shape       # torch.Size([3, 4, 5])
tensor.size()      # Same as shape
tensor.ndim        # 3 (number of dimensions)
tensor.numel()     # 60 (total elements)
tensor.dtype       # torch.float32
tensor.device      # cpu or cuda
```

---

## Data Types in PyTorch

```python
# Integer types
torch.tensor([1, 2, 3], dtype=torch.int32)
torch.tensor([1, 2, 3], dtype=torch.int64)  # Default int

# Float types
torch.tensor([1.0, 2.0], dtype=torch.float32)  # Most common
torch.tensor([1.0, 2.0], dtype=torch.float64)

# Convert types
tensor = torch.tensor([1, 2, 3])
tensor_float = tensor.float()  # int → float
tensor_int = tensor_float.int()  # float → int
```

**For deep learning, use `float32`** (balance of precision and speed)

---

## Creating Tensors Like Other Tensors

```python
x = torch.randn(3, 4)

# Create same shape
torch.zeros_like(x)   # Zeros with same shape as x
torch.ones_like(x)    # Ones with same shape
torch.rand_like(x)    # Random with same shape
```

---

## GPU Support (Preview)

```python
# Check if GPU available
print(torch.cuda.is_available())

# Move tensor to GPU
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    # or
    tensor = tensor.cuda()

# Move back to CPU
tensor = tensor.to('cpu')
# or
tensor = tensor.cpu()
```

We'll cover GPU training in later steps!

---

## Common Operations

### Reshape

```python
tensor = torch.arange(12)
reshaped = tensor.reshape(3, 4)  # 3×4 matrix
# or
reshaped = tensor.view(3, 4)  # Similar but stricter
```

### Concatenation

```python
a = torch.tensor([[1, 2]])
b = torch.tensor([[3, 4]])

torch.cat([a, b], dim=0)  # Stack vertically → (2, 2)
torch.cat([a, b], dim=1)  # Stack horizontally → (1, 4)
```

### Indexing

```python
tensor = torch.arange(12).reshape(3, 4)

tensor[0]        # First row
tensor[:, 1]     # Second column
tensor[1, 2]     # Element at row 1, col 2
tensor[0:2, 1:3] # Slice: rows 0-1, cols 1-2
```

---

## Your First Neural Network Computation

```python
# Input: 1 sample, 10 features
x = torch.randn(1, 10)

# Weights: 10 inputs → 5 outputs
W = torch.randn(10, 5)
b = torch.randn(5)

# Linear transformation (y = xW + b)
y = x @ W + b  # Shape: (1, 5)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
```

**This is the core operation in neural networks!**

---

## Complete the Assignment

Open the Colab notebook and:
1. Create various tensors
2. Convert between NumPy and PyTorch
3. Perform tensor operations

The verification checks your **outputs**, not variable names!

---

## Next: Autograd

In Step 2, you'll learn:
- Automatic differentiation
- Computing gradients
- Understanding backpropagation
- Building blocks of training
