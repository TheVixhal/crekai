# Step 1: Understanding Transformers and GPT Architecture

Build GPT (Generative Pre-trained Transformer) from scratch! Understand how ChatGPT, GPT-4, and LLaMA work.

## What is GPT?

**GPT** = Generative Pre-trained Transformer

- **Generative**: Creates text, one token at a time
- **Pre-trained**: Trained on massive text data
- **Transformer**: Uses self-attention mechanism

**Powers:** ChatGPT, GitHub Copilot, GPT-4, and more!

---

## The Big Picture

### Traditional RNNs
```
Process sequentially: word1 → word2 → word3 →...
Problem: Slow, forgets long context
```

### Transformers (GPT)
```
Process in parallel: [word1, word2, word3, ...] → all at once!
Benefit: Fast, remembers everything via attention
```

---

## Your Assignment

### Task 1: Tokenization Basics

Understand how text becomes numbers:

```python
# Simple character-level tokenizer
text = "hello world"

# Create vocabulary
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode text to numbers
encoded = [char_to_idx[ch] for ch in text]
print(f"'{text}' → {encoded}")

# Decode numbers to text
decoded = ''.join([idx_to_char[i] for i in encoded])
print(f"{encoded} → '{decoded}'")
```

**Expected output:** Encoded and decoded text matching original

---

### Task 2: Position Embeddings

Implement positional encoding:

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    """
    Create positional encodings
    seq_len: sequence length
    d_model: embedding dimension
    """
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                         (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Create for 10 positions, 64 dimensions
pe = positional_encoding(10, 64)
print(f"Positional encoding shape: {pe.shape}")
```

**Expected output:** Tensor of shape (10, 64)

---

### Task 3: Attention Intuition

Understand attention with a simple example:

```python
# Query: "What am I looking for?"
# Key: "What do I contain?"
# Value: "What do I output?"

def simple_attention(query, keys, values):
    """
    Simplified attention mechanism
    """
    # Compute attention scores (dot product)
    scores = np.dot(keys, query)
    
    # Softmax to get weights
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Weighted sum of values
    output = np.sum(weights[:, None] * values, axis=0)
    
    return output, weights

# Example
query = np.array([1.0, 0.5])
keys = np.array([[1.0, 0.3], 
                 [0.2, 0.8],
                 [0.9, 0.2]])
values = np.array([[1, 2], 
                   [3, 4],
                   [5, 6]])

output, weights = simple_attention(query, keys, values)
print(f"Attention weights: {weights}")
print(f"Output: {output}")
```

**Expected output:** Attention weights and weighted output

---

## GPT Architecture Overview

```python
GPT Model:
    ├── Token Embedding (vocab_size → d_model)
    ├── Position Embedding (max_len → d_model)
    ├── Transformer Blocks (×12 or ×24)
    │   ├── Multi-Head Attention
    │   ├── Layer Norm
    │   ├── Feed Forward Network
    │   └── Layer Norm
    └── Output Head (d_model → vocab_size)
```

---

## Key Components We'll Build

### 1. Self-Attention
- Each token attends to all previous tokens
- Learns relationships between words

### 2. Multi-Head Attention
- Multiple attention mechanisms in parallel
- Captures different patterns

### 3. Feed-Forward Network
- 2-layer MLP after attention
- Adds processing capacity

### 4. Layer Normalization
- Stabilizes training
- Faster convergence

### 5. Residual Connections
- Skip connections
- Helps gradient flow

---

## Transformer vs RNN

| Feature | RNN | Transformer |
|---------|-----|-------------|
| **Processing** | Sequential | Parallel |
| **Speed** | Slow | Fast |
| **Long context** | Forgets | Remembers |
| **Training** | Difficult | Easier |
| **Use case** | Old models | Modern (GPT, BERT) |

---

## What Makes GPT Special?

### 1. Causal Attention
```
When predicting word 5, can only see words 1-4
(Can't peek at future words!)
```

### 2. Massive Scale
```
GPT-3: 175 billion parameters
GPT-4: Estimated 1+ trillion parameters
```

### 3. Next Token Prediction
```
Input: "The cat sat on the"
Output: "mat" (most likely next word)
```

---

## Real-World Applications

- 💬 **ChatGPT** - Conversational AI
- 👨‍💻 **GitHub Copilot** - Code generation
- ✍️ **Jasper** - Content writing
- 🎨 **Midjourney** - Image generation (uses text encoder)
- 🔊 **Whisper** - Speech recognition

All use transformer architecture!

---

## Complete the Assignment

Build the foundational components in Colab:
1. Tokenization system
2. Positional encodings
3. Basic attention mechanism

---

## Next Steps

Step 2: Implement self-attention  
Step 3: Multi-head attention  
Step 4: Transformer block  
Step 5: Complete GPT model  
Step 6: Training on text  
Step 7: Text generation  
Step 8: Fine-tuning

Let's build GPT! 🚀
