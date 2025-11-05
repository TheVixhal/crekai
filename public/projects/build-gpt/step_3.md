# Step 3: Multi-Head Attention

Learn why GPT uses multiple attention heads and how to implement them!

## What is Multi-Head Attention?

Instead of one attention mechanism, use **multiple in parallel**:
- Each head learns different patterns
- Head 1: Grammar relationships
- Head 2: Semantic meaning
- Head 3: Long-range dependencies

**More heads = richer representations!**

---

## Your Assignment

### Task 1: Implement Multi-Head Attention

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """Split into multiple heads"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, x, mask=None):
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into heads
        Q = self.split_heads(Q)  # (batch, heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention

# Test
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10
output, attention = mha(x)
print(f"Output shape: {output.shape}")  # (2, 10, 512)
```

**Expected output:** Same shape as input

---

## Understanding Multi-Head

### Why Multiple Heads?

```
Single Head: 
  Can only learn ONE type of relationship

8 Heads:
  Head 1: Subject-verb relationships
  Head 2: Adjective-noun relationships  
  Head 3: Long-range dependencies
  Head 4: Positional patterns
  ... more patterns!
```

### The Math

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₙ) × Wₒ

where headᵢ = Attention(Q×Wᵢq, K×Wᵢk, V×Wᵢv)
```

---

## Complete the Assignment

Implement multi-head attention in Colab!

---

## Next: Transformer Block

Step 4: Combine attention with feed-forward networks.
