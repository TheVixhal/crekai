# Step 7: Text Generation with GPT

Generate text using your trained GPT model!

## Your Assignment

### Task 1: Greedy Decoding

```python
def generate(model, idx, max_new_tokens):
    """
    Generate tokens greedily (pick highest probability)
    idx: starting tokens (batch, seq_len)
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop to max context length
        idx_cond = idx[:, -model.max_len:] if idx.size(1) > model.max_len else idx
        
        # Forward pass
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # Last position only
            
            # Greedy: pick highest probability
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)
    
    return idx

# Generate
start_tokens = torch.tensor([[char_to_idx['T']]], dtype=torch.long)
generated = generate(model, start_tokens, max_new_tokens=100)

# Decode
text = ''.join([idx_to_char[i] for i in generated[0].tolist()])
print(text)
```

**Expected output:** Generated text

---

### Task 2: Sampling with Temperature

```python
def generate_with_temperature(model, idx, max_new_tokens, temperature=1.0):
    """
    Generate with temperature (controls randomness)
    temperature = 1.0: normal sampling
    temperature < 1.0: more conservative (less random)
    temperature > 1.0: more creative (more random)
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_len:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_token], dim=1)
    
    return idx

# Generate with different temperatures
low_temp = generate_with_temperature(model, start_tokens, 100, temperature=0.5)
high_temp = generate_with_temperature(model, start_tokens, 100, temperature=1.5)
```

**Expected output:** Different text styles

---

### Task 3: Top-k and Top-p Sampling

```python
def generate_topk(model, idx, max_new_tokens, k=10):
    """
    Sample from top-k most likely tokens
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_len:]
        
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
            
            # Get top-k logits
            top_k_logits, top_k_indices = torch.topk(logits, k)
            
            # Sample from top-k
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token_idx)
            
            idx = torch.cat([idx, next_token], dim=1)
    
    return idx
```

**Expected output:** Higher quality text

---

## Sampling Strategies

### Greedy (Deterministic)
- Always picks highest probability
- Repetitive, boring output

### Temperature Sampling
- Low temp (0.5): Conservative, coherent
- High temp (1.5): Creative, chaotic

### Top-k Sampling
- Only consider top k tokens
- Prevents very unlikely words

### Top-p (Nucleus) Sampling
- Consider tokens until cumulative prob > p
- Dynamic vocabulary size

---

## Complete the Assignment

Implement text generation with different strategies!

---

## Next: Fine-Tuning

Step 8: Fine-tune GPT for specific tasks.
