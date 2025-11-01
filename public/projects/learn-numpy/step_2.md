# Step 2: Array Operations

Now that you understand the basics, let's explore common array operations.

## Array Shape and Size

\`\`\`python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr.shape)  # (5,)
print(arr.size)   # 5
print(arr.dtype)  # int64
\`\`\`

## Multi-dimensional Arrays

\`\`\`python
# Create a 2D array
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)
\`\`\`

## Task

Create a 2D array with shape (3, 3) containing numbers 1-9. Print its shape and size.

**Hint:** You can use `np.arange()` to create a sequence and reshape it.

Once completed, mark this step as complete!
