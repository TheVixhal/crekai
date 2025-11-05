# Step 4: Complete Transformer Block

Build a full transformer block with attention, feed-forward, and residual connections!

## Your Assignment

### Task 1: Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

---

### Task 2: Layer Normalization

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

---

### Task 3: Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual
        attn_output, _ = self.attention(x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer norm
        
        # Feed-forward with residual
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer norm
        
        return x

# Test
block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)
output = block(x)
print(f"Output shape: {output.shape}")
```

**Expected output:** Same shape as input (2, 10, 512)

---

## Complete the Assignment

Implement the full transformer block!

---

## Next: GPT Model

Step 5: Combine everything into complete GPT!
