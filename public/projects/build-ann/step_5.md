# Step 5: Model Optimization and Evaluation

Improve your network's performance with advanced techniques!

## Topics Covered

- Different optimizers (SGD, Momentum, Adam)
- Learning rate scheduling
- Regularization (L2, Dropout)
- Batch normalization
- Model evaluation metrics

## Key Techniques

### 1. Momentum
```python
# Helps escape local minima
velocity = 0.9 * velocity + learning_rate * gradient
weights -= velocity
```

### 2. Learning Rate Decay
```python
learning_rate = initial_lr * (0.95 ** epoch)
```

### 3. L2 Regularization
```python
# Add to loss
loss += (lambda_reg / 2) * np.sum(weights ** 2)
```

---

## Assignment

Implement momentum optimizer and achieve >95% accuracy on MNIST.

---

## Next: Deployment

Step 6 covers saving and using models in production.
