# Step 2: Self-Attention Mechanism

Implement the core of transformers: **scaled dot-product attention**!

## What is Self-Attention?

Attention allows each word to look at all other words and decide which ones are important.

**Example:**
```
"The cat sat on the mat because it was tired"
                              ↑
When processing "it", attention focuses on "cat" (not "mat")
```

---

## Your Assignment

### Task 1: Implement Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: Query (batch, seq_len, d_k)
    K: Key (batch, seq_len, d_k)
    V: Value (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask (for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Test
seq_len, d_model = 5, 64
Q = torch.randn(1, seq_len, d_model)
K = torch.randn(1, seq_len, d_model)
V = torch.randn(1, seq_len, d_model)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # (1, 5, 64)
```

**Expected output:** Attention output and weights

---

### Task 2: Causal Mask for GPT

```python
def create_causal_mask(seq_len):
    """
    Create mask so position i can only attend to positions <= i
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# Test
mask = create_causal_mask(5)
print(mask)
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]
```

**Expected output:** Lower triangular mask

---

### Task 3: Self-Attention with Mask

```python
# Apply causal attention
seq_len = 5
mask = create_causal_mask(seq_len).unsqueeze(0)  # Add batch dim

output, weights = scaled_dot_product_attention(Q, K, V, mask)

# Verify causality
print("Attention weights (row i attends to columns <= i):")
print(weights[0])
```

**Expected output:** Attention weights respecting causal constraint

---

## Understanding the Math

### Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

- **QK^T**: How much each position attends to others
- **/√d_k**: Scale for numerical stability
- **softmax**: Convert to probabilities
- **×V**: Weighted sum of values

---

## Why Scaling Matters

```python
# Without scaling
scores = Q @ K.T  # Values can be very large

# With scaling
scores = (Q @ K.T) / sqrt(d_k)  # Normalized range

# Large scores → softmax saturates → bad gradients
```

---

## Complete the Assignment

Implement scaled dot-product attention with causal masking in Colab!

---

## Next: Multi-Head Attention

Step 3: Learn to use multiple attention heads in parallel.
