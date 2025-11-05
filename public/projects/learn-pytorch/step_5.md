# Step 5: Real Data with DataLoaders

Train on **real datasets** using PyTorch's DataLoader - the professional way to handle data!

## What You'll Learn

- Loading datasets (MNIST, CIFAR10)
- DataLoader for batching
- Data augmentation
- Train/validation splits

---

## Your Assignment

### Task 1: Load MNIST Dataset

```python
from torchvision import datasets, transforms

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

**Expected output:** 60,000 training, 10,000 test samples

---

### Task 2: Create DataLoaders

```python
from torch.utils.data import DataLoader

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True  # Shuffle for training
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False  # Don't shuffle test data
)

# Iterate through batches
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
    break  # Just show first batch
```

**Expected output:** Batches of shape (64, 1, 28, 28)

---

### Task 3: Train on Real Data

Train a CNN on MNIST for 5 epochs:

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(5):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**Expected output:** Trained model with >90% accuracy

---

## DataLoader Features

```python
DataLoader(
    dataset,
    batch_size=32,      # Samples per batch
    shuffle=True,       # Shuffle each epoch
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    drop_last=False     # Drop incomplete last batch?
)
```

---

## Data Transforms

```python
from torchvision import transforms

# Common transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

---

## Evaluation

```python
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Test
accuracy = evaluate(model, test_loader)
print(f"Accuracy: {accuracy:.2f}%")
```

---

## Complete Training Pipeline

```python
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training
for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    val_acc = evaluate(model, test_loader)
    print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={val_acc:.2f}%')
```

---

## Congratulations! 🎉

You've completed PyTorch fundamentals! You can now:
- ✅ Create tensors and operations
- ✅ Use autograd for gradients
- ✅ Build neural networks
- ✅ Train on real datasets
- ✅ Evaluate model performance

**You're ready to build real AI applications!**

---

## Next Steps

Continue with:
- **Build ANN** - Create custom architectures
- **Build GPT** - Implement transformers
- Explore CNNs, RNNs, and advanced topics!
