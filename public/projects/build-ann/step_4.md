# Step 4: Training Loop and Optimization

Train your network on MNIST and watch it learn!

## Your Assignment

### Task 1: Load and Preprocess MNIST

```python
from sklearn.datasets import fetch_openml

# Load MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X.values / 255.0  # Normalize to 0-1
y = y.astype(int)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# One-hot encode labels
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot(y_train.values)
y_test_onehot = one_hot(y_test.values)
```

---

### Task 2: Complete Training Loop

```python
# Initialize model
model = NeuralNetwork([784, 128, 64, 10])

# Training parameters
epochs = 50
batch_size = 128
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train_onehot[indices]
    
    epoch_loss = 0
    batches = len(X_train) // batch_size
    
    # Mini-batch training
    for i in range(batches):
        start = i * batch_size
        end = start + batch_size
        
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        
        # Train on batch
        loss = model.train_step(X_batch, y_batch, learning_rate)
        epoch_loss += loss
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / batches
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
```

---

### Task 3: Evaluate Model

```python
def predict(model, X):
    predictions, _ = model.forward(X)
    return np.argmax(predictions, axis=1)

def accuracy(model, X, y):
    preds = predict(model, X)
    return np.mean(preds == y) * 100

# Test accuracy
train_acc = accuracy(model, X_train, y_train)
test_acc = accuracy(model, X_test, y_test)

print(f'Train Accuracy: {train_acc:.2f}%')
print(f'Test Accuracy: {test_acc:.2f}%')
```

**Expected output:** >90% test accuracy

---

## Complete the Assignment

Train a full network on MNIST from scratch!

---

## Next: Advanced Topics

Steps 5-7 cover optimizations and deployment.
