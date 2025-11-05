# Step 1: Understanding Neural Networks

Build your first Artificial Neural Network (ANN) from scratch! Learn how the magic works under the hood.

## What You'll Build

A complete neural network that can:
- Classify handwritten digits (MNIST)
- Learn from data
- Achieve 95%+ accuracy

**All from scratch** - no black boxes!

---

## What is an Artificial Neural Network?

### The Biological Inspiration

```
Biological Neuron:
Dendrites → Cell Body → Axon → Synapses

Artificial Neuron:
Inputs → Weighted Sum → Activation → Output
```

### Mathematical Definition

```python
# Single neuron
output = activation(sum(inputs * weights) + bias)

# For input [x1, x2, x3]:
z = w1*x1 + w2*x2 + w3*x3 + b
output = activation(z)
```

---

## Your Assignment

### Task 1: Implement a Single Neuron

Create a function that computes a neuron's output:

```python
import numpy as np

def neuron(inputs, weights, bias):
    """
    Single neuron computation
    inputs: array of input values
    weights: array of weights
    bias: bias value
    """
    # Weighted sum
    z = np.dot(inputs, weights) + bias
    
    # ReLU activation
    output = max(0, z)
    
    return output

# Test
inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, 0.3, 0.2])
bias = 0.1

result = neuron(inputs, weights, bias)
print(f"Neuron output: {result}")
```

**Expected output:** Single number (neuron's activation)

---

### Task 2: Activation Functions

Implement common activation functions:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum()

# Test them
x = np.array([-2, -1, 0, 1, 2])
print("ReLU:", relu(x))
print("Sigmoid:", sigmoid(x))
```

**Expected output:** Transformed arrays

---

### Task 3: Layer of Neurons

Implement a full layer (multiple neurons):

```python
def dense_layer(inputs, weights, biases, activation='relu'):
    """
    Dense layer: multiple neurons
    inputs: (n_features,)
    weights: (n_features, n_neurons)
    biases: (n_neurons,)
    """
    # Matrix multiplication
    z = np.dot(inputs, weights) + biases
    
    # Activation
    if activation == 'relu':
        return relu(z)
    elif activation == 'sigmoid':
        return sigmoid(z)
    return z

# Test: 3 inputs → 4 neurons
inputs = np.array([1.0, 2.0, 3.0])
weights = np.random.randn(3, 4)
biases = np.random.randn(4)

output = dense_layer(inputs, weights, biases)
print(f"Layer output shape: {output.shape}")  # (4,)
```

**Expected output:** Array with 4 values

---

## Network Architecture

### Simple 3-Layer Network

```
Input Layer (784) 
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Output Layer (10 neurons, Softmax)
```

**For MNIST:**
- Input: 28×28 = 784 pixels
- Output: 10 classes (digits 0-9)

---

## Forward Propagation

```python
def forward_pass(x):
    # Layer 1: 784 → 128
    h1 = dense_layer(x, W1, b1, 'relu')
    
    # Layer 2: 128 → 64
    h2 = dense_layer(h1, W2, b2, 'relu')
    
    # Output: 64 → 10
    output = dense_layer(h2, W3, b3, 'softmax')
    
    return output

# Prediction
prediction = forward_pass(image_pixels)
predicted_digit = np.argmax(prediction)
```

---

## Why Each Component?

### Weights
- **Learn patterns** from data
- Adjusted during training
- Different for each connection

### Biases
- **Shift activation** threshold
- One per neuron
- Like y-intercept in y=mx+b

### Activations
- **Add non-linearity** (make network powerful)
- Without them: network = linear regression
- ReLU is most common: simple and effective

### Multiple Layers
- **Learn complex features**
- Layer 1: edges
- Layer 2: shapes
- Layer 3: objects

---

## The Learning Process (Preview)

```python
for epoch in range(num_epochs):
    for batch in data:
        # 1. Forward pass
        predictions = model(batch_x)
        
        # 2. Calculate loss
        loss = compute_loss(predictions, batch_y)
        
        # 3. Backward pass (compute gradients)
        gradients = compute_gradients(loss)
        
        # 4. Update weights
        weights -= learning_rate * gradients
```

We'll implement this in the next steps!

---

## Key Concepts to Remember

🎯 **Neuron** = Weighted sum + Activation  
🎯 **Layer** = Collection of neurons  
🎯 **Network** = Stack of layers  
🎯 **Forward pass** = Input → Output  
🎯 **Backward pass** = Compute gradients (next step!)  

---

## Complete the Assignment

Implement the basic building blocks in Colab:
1. Single neuron
2. Activation functions
3. Full layer

These are the **foundations** of all neural networks!

---

## Next: Backpropagation

In Step 2, you'll learn:
- How networks learn
- Computing gradients
- Implementing backpropagation
- The chain rule in action
