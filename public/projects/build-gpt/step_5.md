# Step 5: Complete GPT Model

Assemble all components into a working GPT model!

## Your Assignment

### Task 1: Build Complete GPT

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, max_len=512):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model*4)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, idx):
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)
        
        # Position embeddings
        pos = torch.arange(0, seq_len, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Combine
        x = tok_emb + pos_emb
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device))
        
        # Through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Layer norm and output
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# Create mini GPT
vocab_size = 10000
model = GPT(vocab_size, d_model=256, num_heads=8, num_layers=6)

# Test
input_ids = torch.randint(0, vocab_size, (2, 20))  # 2 samples, 20 tokens
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # (2, 20, 10000)
```

**Expected output:** Logits for next token prediction

---

## Complete the Assignment

Build the complete GPT architecture!

---

## Next: Training

Step 6: Train GPT on text data.
