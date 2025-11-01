# Step 3: Array Indexing and Slicing

Master the art of accessing and manipulating array elements.

## Basic Indexing

\`\`\`python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print(arr[0])   # 10
print(arr[-1])  # 50
\`\`\`

## Slicing

\`\`\`python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(arr[2:5])      # [3 4 5]
print(arr[::2])      # [1 3 5 7 9] (every 2nd element)
print(arr[::-1])     # [10 9 8 7 6 5 4 3 2 1] (reversed)
\`\`\`

## Task

Create an array with numbers 1-20. Extract every 3rd element starting from index 2.

Once completed, mark this step as complete!
