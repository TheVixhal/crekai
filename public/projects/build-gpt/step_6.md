# Step 6: Training GPT on Text Data

Train your GPT model to generate text!

## Your Assignment

### Task 1: Prepare Text Dataset

```python
# Load text data
with open('shakespeare.txt', 'r') as f:
    text = f.read()

# Create vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode text
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# Create training samples
def get_batch(data, seq_len, batch_size):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# Test
x, y = get_batch(data, seq_len=128, batch_size=32)
print(f"Batch shapes: {x.shape}, {y.shape}")
```

---

### Task 2: Training Loop

```python
# Initialize
model = GPT(vocab_size, d_model=256, num_heads=8, num_layers=6)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training
for epoch in range(100):
    # Get batch
    x_batch, y_batch = get_batch(data, seq_len=128, batch_size=32)
    
    # Forward
    logits = model(x_batch)
    
    # Reshape for loss
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = y_batch.view(B*T)
    
    # Compute loss
    loss = F.cross_entropy(logits, targets)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

**Expected output:** Decreasing loss over epochs

---

## Complete the Assignment

Train GPT on text data!

---

## Next: Text Generation

Step 7: Generate text with your trained model.
