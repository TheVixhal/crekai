# Step 2: Backpropagation from Scratch

Implement the algorithm that makes neural networks learn: **backpropagation**!

## What is Backpropagation?

**Backpropagation** = Computing gradients by going backwards through the network using the chain rule.

```
Forward: Input → Layer 1 → Layer 2 → Output → Loss
Backward: Loss → ∂L/∂W3 → ∂L/∂W2 → ∂L/∂W1
```

---

## Your Assignment

### Task 1: Compute Loss Gradient

Implement MSE loss and its gradient:

```python
def mse_loss(predictions, targets):
    """Mean Squared Error"""
    return np.mean((predictions - targets) ** 2)

def mse_gradient(predictions, targets):
    """Gradient of MSE: 2(y_pred - y_true) / n"""
    n = len(predictions)
    return 2 * (predictions - targets) / n

# Test
y_pred = np.array([2.5, 3.0, 1.5])
y_true = np.array([3.0, 3.0, 2.0])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)

print(f"Loss: {loss}")
print(f"Gradient: {grad}")
```

**Expected output:** Loss value and gradient array

---

### Task 2: ReLU Backward

Implement ReLU activation and its derivative:

```python
def relu(x):
    return np.maximum(0, x)

def relu_gradient(x):
    """
    Derivative of ReLU:
    1 if x > 0, else 0
    """
    return (x > 0).astype(float)

# Test
x = np.array([-2, -1, 0, 1, 2])
output = relu(x)           # [0, 0, 0, 1, 2]
grad = relu_gradient(x)    # [0, 0, 0, 1, 1]
```

**Expected output:** ReLU output and gradient

---

### Task 3: Backward Pass Through One Layer

```python
def layer_backward(d_output, inputs, weights, z):
    """
    Backward pass through dense layer
    d_output: gradient from next layer
    inputs: inputs to this layer
    weights: layer weights
    z: pre-activation values
    """
    # Gradient of activation
    d_activation = relu_gradient(z)
    
    # Gradient at pre-activation
    d_z = d_output * d_activation
    
    # Gradients for weights and biases
    d_weights = np.outer(inputs, d_z)
    d_biases = d_z
    
    # Gradient to pass to previous layer
    d_inputs = np.dot(weights, d_z)
    
    return d_weights, d_biases, d_inputs

# Test with dummy data
inputs = np.array([1.0, 2.0, 3.0])
weights = np.random.randn(3, 4)
biases = np.random.randn(4)
z = np.dot(inputs, weights) + biases
d_output = np.random.randn(4)

d_w, d_b, d_in = layer_backward(d_output, inputs, weights, z)
```

**Expected output:** Gradients for weights, biases, and inputs

---

## The Chain Rule

### Simple Example

```
y = f(g(x))
dy/dx = df/dg × dg/dx
```

### In Neural Networks

```
Loss = L(f₃(f₂(f₁(x))))

∂L/∂W₁ = ∂L/∂f₃ × ∂f₃/∂f₂ × ∂f₂/∂f₁ × ∂f₁/∂W₁
```

Backprop computes this efficiently!

---

## Full Backward Pass

```python
def backward_pass(x, y_true, W1, b1, W2, b2, h1, output):
    """
    Complete backward pass for 2-layer network
    """
    # Output layer gradient
    d_output = mse_gradient(output, y_true)
    
    # Layer 2 gradients
    d_W2 = np.outer(h1, d_output)
    d_b2 = d_output
    d_h1 = np.dot(W2.T, d_output)
    
    # Layer 1 gradients (through ReLU)
    d_h1_activated = d_h1 * relu_gradient(h1)
    d_W1 = np.outer(x, d_h1_activated)
    d_b1 = d_h1_activated
    
    return d_W1, d_b1, d_W2, d_b2
```

---

## Gradient Descent Update

```python
# After computing gradients
learning_rate = 0.01

W1 -= learning_rate * d_W1
b1 -= learning_rate * d_b1
W2 -= learning_rate * d_W2
b2 -= learning_rate * d_b2
```

**This is how the network learns!**

---

## Complete Example: Training One Step

```python
# Initialize
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([0, 1])  # One-hot encoded

W1 = np.random.randn(3, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2)
b2 = np.random.randn(2)

# Forward pass
z1 = np.dot(x, W1) + b1
h1 = relu(z1)
z2 = np.dot(h1, W2) + b2
output = softmax(z2)

# Compute loss
loss = -np.sum(y_true * np.log(output + 1e-8))

# Backward pass
d_W1, d_b1, d_W2, d_b2 = backward_pass(...)

# Update
W1 -= 0.01 * d_W1
# ... update others
```

---

## Complete the Assignment

Implement backpropagation in Colab:
1. Loss gradients
2. Activation gradients  
3. Full backward pass

Understanding this makes you a **true deep learning practitioner**!

---

## Next: Complete Implementation

Step 3: Build the full training loop with NumPy

