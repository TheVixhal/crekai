# Step 3: Complete Network Implementation with NumPy

Build a complete neural network from scratch using only NumPy - no frameworks!

## What You'll Build

A full 3-layer neural network that can classify MNIST digits with 95%+ accuracy.

---

## Your Assignment

### Task 1: Initialize Network Weights

```python
import numpy as np

def initialize_weights(layer_dims):
    """
    Initialize weights for all layers
    layer_dims: list of layer sizes [784, 128, 64, 10]
    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        # He initialization for ReLU
        parameters[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(2/layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((1, layer_dims[l]))
    
    return parameters

# Initialize network: 784 → 128 → 64 → 10
params = initialize_weights([784, 128, 64, 10])
```

**Expected output:** Dictionary with weights W1, b1, W2, b2, W3, b3

---

### Task 2: Complete Forward Pass

```python
def forward_propagation(X, parameters):
    """
    Forward pass through entire network
    X: input data (batch_size, 784)
    """
    cache = {'A0': X}
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        # Linear transformation
        Z = np.dot(A, parameters[f'W{l}']) + parameters[f'b{l}']
        # ReLU activation
        A = np.maximum(0, Z)
        # Cache for backprop
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    
    # Output layer (no activation yet)
    Z_out = np.dot(A, parameters[f'W{L}']) + parameters[f'b{L}']
    # Softmax
    exp_scores = np.exp(Z_out - np.max(Z_out, axis=1, keepdims=True))
    A_out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    cache[f'Z{L}'] = Z_out
    cache[f'A{L}'] = A_out
    
    return A_out, cache

# Test
X = np.random.randn(5, 784)  # 5 samples
output, cache = forward_propagation(X, params)
print(f"Output shape: {output.shape}")  # (5, 10)
print(f"Sum per row: {output.sum(axis=1)}")  # Should be ~1.0 (probabilities)
```

**Expected output:** (batch_size, 10) predictions summing to 1.0

---

### Task 3: Complete Backward Pass

```python
def backward_propagation(AL, Y, cache, parameters):
    """
    Backward pass through entire network
    AL: output predictions
    Y: true labels (one-hot)
    """
    gradients = {}
    m = AL.shape[0]  # batch size
    L = len(parameters) // 2
    
    # Output layer gradient
    dZ = AL - Y
    
    # Backward through layers
    for l in reversed(range(1, L + 1)):
        A_prev = cache[f'A{l-1}']
        
        # Gradients
        gradients[f'dW{l}'] = (1/m) * np.dot(A_prev.T, dZ)
        gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        if l > 1:
            # Gradient for previous layer
            dA_prev = np.dot(dZ, parameters[f'W{l}'].T)
            # Through ReLU
            dZ = dA_prev * (cache[f'Z{l-1}'] > 0)
    
    return gradients

# Test
Y = np.eye(10)[np.random.randint(0, 10, 5)]  # One-hot labels
grads = backward_propagation(output, Y, cache, params)
```

**Expected output:** Gradients for all weights and biases

---

## Complete Network Class

```python
class NeuralNetwork:
    def __init__(self, layer_dims):
        self.parameters = initialize_weights(layer_dims)
    
    def forward(self, X):
        return forward_propagation(X, self.parameters)
    
    def backward(self, AL, Y, cache):
        return backward_propagation(AL, Y, cache, self.parameters)
    
    def update_parameters(self, gradients, learning_rate):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    
    def train_step(self, X, Y, learning_rate):
        # Forward
        AL, cache = self.forward(X)
        # Backward
        grads = self.backward(AL, Y, cache)
        # Update
        self.update_parameters(grads, learning_rate)
        # Compute loss
        loss = -np.mean(Y * np.log(AL + 1e-8))
        return loss

# Create network
model = NeuralNetwork([784, 128, 64, 10])
```

---

## Complete the Assignment

Implement the full network in Colab with all components working together!

---

## Next: Training Loop

Step 4 covers training the network on real data.
