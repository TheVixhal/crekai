# Step 3: Building Neural Networks with nn.Module

Learn to build neural networks using PyTorch's **nn.Module** - the foundation of all PyTorch models!

## What is nn.Module?

`nn.Module` is the base class for all neural network components. Every model you build will inherit from it.

```python
import torch
import torch.nn as nn

class MyFirstModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here
        
    def forward(self, x):
        # Define forward pass
        return x
```

---

## Your Assignment

### Task 1: Build a Simple Linear Layer

Create a single linear layer that transforms 10 inputs to 5 outputs:

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # 10 inputs → 5 outputs
    
    def forward(self, x):
        return self.linear(x)

# Create model
model = SimpleModel()

# Test it
x = torch.randn(1, 10)  # 1 sample, 10 features
output = model(x)       # Shape: (1, 5)
```

**Expected output:** Model that produces 5 outputs from 10 inputs

---

### Task 2: Multi-Layer Network

Build a 3-layer network: 784 → 128 → 64 → 10

```python
class MultiLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No ReLU on output
        return x

model = MultiLayerNet()
```

**Expected output:** 3-layer network with ReLU activations

---

### Task 3: Activation Functions

Create a model and test different activation functions:

```python
class ActivationDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x, activation='relu'):
        x = self.linear(x)
        
        if activation == 'relu':
            return torch.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'tanh':
            return torch.tanh(x)
        return x

model = ActivationDemo()
x = torch.randn(1, 10)

out_relu = model(x, 'relu')
out_sigmoid = model(x, 'sigmoid')
```

**Expected output:** Different outputs for each activation

---

## Common Layers

### Linear (Fully Connected)

```python
nn.Linear(in_features, out_features)

# Example: 100 → 50
layer = nn.Linear(100, 50)
```

### Activation Functions

```python
nn.ReLU()         # Most common: f(x) = max(0, x)
nn.Sigmoid()      # f(x) = 1 / (1 + e^-x)
nn.Tanh()         # f(x) = tanh(x)
nn.LeakyReLU()    # Like ReLU but small negative slope
nn.Softmax(dim=1) # For classification output
```

### Dropout (Regularization)

```python
nn.Dropout(p=0.5)  # Randomly drop 50% of neurons
```

### Batch Normalization

```python
nn.BatchNorm1d(num_features)  # For 1D data
nn.BatchNorm2d(num_channels)  # For images
```

---

## Sequential: Quick Model Building

For simple models, use `nn.Sequential`:

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Forward pass
output = model(input)
```

**When to use:**
- ✅ Simple linear flow
- ❌ Complex architectures (use nn.Module)

---

## Model Parameters

```python
model = SimpleModel()

# See all parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

---

## Forward Pass Deep Dive

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        # This is called when you do: model(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel()
output = model(input)  # Calls forward() automatically
```

---

## Save and Load Models

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode
```

---

## Real Architecture: MNIST Classifier

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 digit classes
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.flatten(x)  # (batch, 28, 28) → (batch, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Logits (no softmax yet)

model = MNISTNet()
```

---

## Complete the Assignment

Build neural network architectures in Colab:
1. Simple linear model
2. Multi-layer network
3. Experiment with activations

**Next: Training loops!**

---

## Next: Training Your First Model

Step 4 covers:
- Loss functions
- Optimizers
- Complete training loop
- Evaluation
