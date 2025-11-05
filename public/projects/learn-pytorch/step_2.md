# Step 2: Autograd - Automatic Differentiation

Learn PyTorch's **killer feature**: automatic gradient computation. This is what makes training neural networks possible!

## What is Autograd?

**Autograd** automatically computes derivatives (gradients) of your operations. No manual calculus needed!

### Why It Matters

Training a neural network means:
1. Make a prediction
2. Calculate error (loss)
3. **Compute gradients** ← Autograd does this!
4. Update weights
5. Repeat

Without autograd, you'd need to manually derive and code every gradient. With PyTorch: **automatic**! 🎉

---

## requires_grad: Tracking Computations

```python
# Regular tensor (no gradient tracking)
x = torch.tensor([2.0, 3.0])

# Tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Or enable later
x = torch.tensor([2.0, 3.0])
x.requires_grad = True
```

---

## Your Assignment

### Task 1: Compute Simple Gradients

Create a tensor, perform operations, compute gradients:

```python
# Create tensor with gradient tracking
x = torch.tensor([3.0], requires_grad=True)

# Perform operation: y = x²
y = x ** 2

# Compute gradient: dy/dx = 2x = 6
y.backward()

# Access gradient
print(x.grad)  # tensor([6.])
```

**Expected output:** Gradient value of 6.0

---

### Task 2: More Complex Function

Compute gradient of: `z = 2x² + 3x + 1` at x = 4

```python
x = torch.tensor([4.0], requires_grad=True)

# Your function
z = 2 * x**2 + 3 * x + 1

# Compute gradient
z.backward()

# dz/dx = 4x + 3 = 4(4) + 3 = 19
print(x.grad)  # Should be 19.0
```

**Expected output:** Gradient = 19.0

---

### Task 3: Vector Gradients

Compute gradients for multiple inputs:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Sum of squares
y = (x ** 2).sum()  # y = x₁² + x₂² + x₃²

# Compute gradients
y.backward()

# dy/dx = [2x₁, 2x₂, 2x₃] = [2, 4, 6]
print(x.grad)
```

**Expected output:** Gradient vector [2.0, 4.0, 6.0]

---

## Understanding backward()

### The Computation Graph

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3      # y = 3x
z = y + 5      # z = 3x + 5
w = z ** 2     # w = (3x + 5)²

w.backward()   # Compute dw/dx

# PyTorch tracks: x → y → z → w
# Then computes gradient backwards: w → z → y → x
```

### Chain Rule in Action

```
w = (3x + 5)²
dw/dx = 2(3x + 5) × 3 = 6(3x + 5)

At x = 2:
dw/dx = 6(6 + 5) = 66
```

PyTorch does this automatically! 🎯

---

## Gradient Accumulation

```python
x = torch.tensor([2.0], requires_grad=True)

# First computation
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])

# Second computation
z = x ** 3
z.backward()
print(x.grad)  # tensor([16.]) ← Accumulated!

# Clear gradients
x.grad.zero_()
```

**Important:** Gradients accumulate by default!

---

## The grad Attribute

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

y.backward()

# Check gradients
print(x.grad)          # The gradient
print(y.grad)          # None (only leaf nodes)
print(x.is_leaf)       # True
print(y.is_leaf)       # False
```

**Only leaf tensors** (inputs) store gradients by default.

---

## Detaching from Graph

Sometimes you don't want to track gradients:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# Detach (stop tracking)
y_detached = y.detach()

# No gradient tracking
z = y_detached + 10
```

### with torch.no_grad()

```python
x = torch.tensor([2.0], requires_grad=True)

with torch.no_grad():
    y = x ** 2  # No gradient tracking
    z = y + 5
```

**Use for:** Evaluation/inference (not training)

---

## Real Training Example

### Simple Linear Model

```python
# Data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model parameters (trainable)
w = torch.tensor([[1.5]], requires_grad=True)
b = torch.tensor([[0.5]], requires_grad=True)

# Forward pass
y_pred = x @ w + b  # y = wx + b

# Loss (error)
loss = ((y_pred - y_true) ** 2).mean()

# Backward pass (compute gradients)
loss.backward()

# Update weights (gradient descent)
with torch.no_grad():
    w -= 0.01 * w.grad  # Learning rate = 0.01
    b -= 0.01 * b.grad
    
    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()
```

**This is the training loop!** We'll build on this in future steps.

---

## Common Patterns

### 1. Training Step

```python
# Zero gradients
optimizer.zero_grad()

# Forward pass
output = model(input)
loss = loss_fn(output, target)

# Backward pass
loss.backward()

# Update weights
optimizer.step()
```

### 2. Validation (No Gradients)

```python
model.eval()
with torch.no_grad():
    output = model(input)
    accuracy = compute_accuracy(output, target)
```

---

## Debugging Gradients

```python
# Check if tensor requires grad
print(x.requires_grad)

# Check gradient value
print(x.grad)

# Check gradient function
print(y.grad_fn)  # Shows operation that created y

# Manually compute gradient
print(torch.autograd.grad(y, x))
```

---

## Complete the Assignment

Practice autograd in Colab:
1. Compute simple gradients
2. Complex function gradients
3. Vector gradients

Understanding autograd is **critical** for deep learning!

---

## Next: Building Neural Networks

In Step 3, you'll learn:
- nn.Module
- Creating custom layers
- Forward pass
- Loss functions
