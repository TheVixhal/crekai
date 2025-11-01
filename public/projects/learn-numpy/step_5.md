# Step 5: Broadcasting and Advanced Operations

Understanding broadcasting is key to NumPy mastery.

## Broadcasting

\`\`\`python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10

print(arr + scalar)   # Scalar is broadcast to each element

# Result:
# [[11 12 13]
#  [14 15 16]]
\`\`\`

## Dot Product

\`\`\`python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))   # 32 (1*4 + 2*5 + 3*6)
\`\`\`

## Task

Create a 3x3 matrix. Multiply it by a scalar value. Compute the dot product of two 1D arrays. Calculate the sum of all elements in the resulting matrix.

Congratulations! You've completed the NumPy fundamentals course!

Once completed, mark this step as complete to finish the project!
