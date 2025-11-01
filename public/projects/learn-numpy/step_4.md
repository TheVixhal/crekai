# Step 4: Mathematical Operations

Learn how to perform mathematical operations on arrays.

## Element-wise Operations

\`\`\`python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)   # [5 7 9]
print(a * b)   # [4 10 18]
print(b - a)   # [3 3 3]
\`\`\`

## Built-in Functions

\`\`\`python
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))      # 15
print(np.mean(arr))     # 3.0
print(np.std(arr))      # 1.41...
print(np.max(arr))      # 5
print(np.min(arr))      # 1
\`\`\`

## Task

Create two arrays of size 5. Perform addition, subtraction, and multiplication. Calculate the mean and standard deviation of their product.

Once completed, mark this step as complete!
