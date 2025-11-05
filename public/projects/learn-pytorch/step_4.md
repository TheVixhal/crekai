# Step 4: Training Your First Model

Put it all together: build a complete training loop and train a real neural network!

## The Training Process

```
1. Forward pass → Make predictions
2. Compute loss → How wrong are we?
3. Backward pass → Calculate gradients
4. Update weights → Improve the model
5. Repeat!
```

---

## Your Assignment

### Task 1: Loss Functions

Create predictions and compute different losses:

```python
# Predictions and targets
predictions = torch.tensor([[2.5, 1.3, 0.8]])
targets = torch.tensor([[3.0, 1.0, 1.0]])

# MSE Loss (regression)
mse = nn.MSELoss()
loss_mse = mse(predictions, targets)

# MAE Loss
mae = nn.L1Loss()
loss_mae = mae(predictions, targets)
```

For classification:

```python
# Logits (raw scores)
logits = torch.tensor([[2.0, 1.0, 0.1]])
# True class (index)
target = torch.tensor([0])

# Cross Entropy Loss
ce = nn.CrossEntropyLoss()
loss_ce = ce(logits, target)
```

**Expected output:** Loss values

---

### Task 2: Optimizers

Create a model and optimizer, perform one update step:

```python
# Simple model
model = nn.Linear(10, 1)

# SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy data
x = torch.randn(5, 10)
y_true = torch.randn(5, 1)

# Forward
y_pred = model(x)
loss = nn.MSELoss()(y_pred, y_true)

# Backward
optimizer.zero_grad()  # Clear old gradients
loss.backward()        # Compute new gradients
optimizer.step()       # Update weights
```

**Expected output:** Model with updated weights

---

### Task 3: Complete Training Loop

Train a model for 10 epochs on dummy data:

```python
# Create model
model = nn.Sequential(
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dummy dataset
X = torch.randn(100, 20)
y = torch.randn(100, 1)

# Training loop
for epoch in range(10):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**Expected output:** Decreasing loss values

---

## Common Optimizers

### SGD (Stochastic Gradient Descent)
```python
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Adam (Adaptive Moment Estimation) ⭐
```python
torch.optim.Adam(model.parameters(), lr=0.001)
# Most popular! Works well out of the box
```

### AdamW (Adam with Weight Decay)
```python
torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# Better for transformers
```

### RMSprop
```python
torch.optim.RMSprop(model.parameters(), lr=0.001)
```

---

## Loss Functions

### Regression

```python
nn.MSELoss()        # Mean Squared Error
nn.L1Loss()         # Mean Absolute Error
nn.SmoothL1Loss()   # Huber loss
```

### Classification

```python
nn.CrossEntropyLoss()  # Multi-class (includes softmax)
nn.BCELoss()           # Binary (needs sigmoid first)
nn.BCEWithLogitsLoss() # Binary (includes sigmoid)
```

---

## Training Best Practices

### 1. Zero Gradients First

```python
# ✅ Correct order
optimizer.zero_grad()
loss.backward()
optimizer.step()

# ❌ Wrong - gradients accumulate!
loss.backward()
optimizer.step()
```

### 2. Batch Processing

```python
# Process in batches, not all data at once
batch_size = 32

for i in range(0, len(X), batch_size):
    batch_x = X[i:i+batch_size]
    batch_y = y[i:i+batch_size]
    
    # Train on batch
    predictions = model(batch_x)
    loss = criterion(predictions, batch_y)
    # ...
```

### 3. Model Modes

```python
model.train()     # Training mode (dropout active)
# ... training ...

model.eval()      # Evaluation mode (dropout off)
# ... validation ...
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Setup
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training
epochs = 100
for epoch in range(epochs):
    # Forward
    output = model(X_train)
    loss = criterion(output, y_train)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

---

## Monitoring Training

```python
# Track losses
train_losses = []

for epoch in range(epochs):
    # ... training code ...
    train_losses.append(loss.item())

# Plot
import matplotlib.pyplot as plt
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

---

## Complete the Assignment

Implement a complete training loop in Colab:
1. Define loss functions
2. Set up optimizers
3. Train for multiple epochs
4. Monitor loss decreasing

---

## Next: Real Datasets

Step 5 covers:
- Loading MNIST
- DataLoaders
- Training on real data
- Evaluation metrics
