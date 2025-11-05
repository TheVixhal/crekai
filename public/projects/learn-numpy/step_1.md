# Step 1: Introduction to NumPy

Welcome to your NumPy learning journey! NumPy (Numerical Python) is the fundamental package for scientific computing in Python and the foundation of modern AI/ML.

## What is NumPy?

NumPy provides:
- **Powerful N-dimensional arrays** - Fast, memory-efficient array objects
- **Mathematical functions** - Vectorized operations for speed
- **Linear algebra tools** - Essential for machine learning
- **Random number generation** - For simulations and ML

## Why NumPy for AI/ML?

Every major AI framework builds on NumPy:
- 🔥 **PyTorch** - Uses NumPy-like tensors
- 🧠 **TensorFlow** - Compatible with NumPy arrays
- 📊 **Pandas** - Built on top of NumPy
- 🤖 **Scikit-learn** - Uses NumPy arrays throughout

**Bottom line:** Master NumPy = Master AI fundamentals

---

## Your First Assignment

### Task 1: Create Your First Array

Create a NumPy array containing the numbers from 1 to 10.

```python
import numpy as np

# Create an array [1, 2, 3, ..., 10]
# You can use: np.array(), np.arange(), or np.linspace()
```

**Expected output:** A 1D array with 10 elements, values from 1 to 10

---

### Task 2: Array Operations

Multiply all elements in your array by 2.

```python
# Multiply your array by 2
# NumPy makes this super easy with vectorization!
```

**Expected output:** Array with values from 2 to 20

---

### Task 3: Statistical Analysis

Calculate the **mean** and **median** of your original array (1 to 10).

```python
# Use NumPy's built-in functions:
# - np.mean()
# - np.median()
```

**Expected output:** 
- Mean = 5.5
- Median = 5.5

---

## Key Concepts

### 1. NumPy Arrays vs Python Lists

```python
# Python list
my_list = [1, 2, 3, 4, 5]

# NumPy array
my_array = np.array([1, 2, 3, 4, 5])
```

**Why NumPy is better:**
- ⚡ 50-100x faster for large datasets
- 💾 Uses less memory
- 🎯 Vectorized operations (no loops needed!)
- 🔧 Built-in mathematical functions

### 2. Vectorization

```python
# Python way (slow)
result = []
for x in my_list:
    result.append(x * 2)

# NumPy way (fast)
result = my_array * 2  # That's it!
```

### 3. Useful Functions

```python
np.arange(1, 11)     # Creates [1, 2, ..., 10]
np.zeros(5)          # Creates [0, 0, 0, 0, 0]
np.ones(5)           # Creates [1, 1, 1, 1, 1]
np.linspace(0, 1, 5) # 5 evenly spaced numbers from 0 to 1
```

---

## Practice in Google Colab

1. **Open the assignment notebook** for this step
2. **Complete the three tasks** above
3. **Run the verification cell** to check your work
4. **Get instant feedback** and unlock the next step!

---

## Tips for Success

✅ **Experiment** - Try different approaches  
✅ **Use help()** - Type `help(np.arange)` to see documentation  
✅ **Check shapes** - Use `.shape` to see array dimensions  
✅ **Print often** - See what your arrays look like  

---

## Next Step Preview

In Step 2, you'll learn about:
- Multi-dimensional arrays (matrices)
- Array indexing and slicing
- Reshaping and transposing

Complete this assignment to unlock! 🚀
