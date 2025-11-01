```python
# Scalar
x = np.array(6)
print("x: ", x)
print("x ndim: ", x.ndim)      # number of dimensions
print("x shape:", x.shape)     # dimensions
print("x size: ", x.size)      # size of elements
print("x dtype: ", x.dtype)    # data type
```
x:  6
x ndim:  0
x shape: ()
x size:  1
x dtype:  int64

---
```python
# Vector
x = np.array([1.3, 2.2, 1.7])
print("x: ", x)
print("x ndim: ", x.ndim)
print("x shape:", x.shape)
print("x size: ", x.size)
print("x dtype: ", x.dtype)    # notice the float datatype
```
x:  [1.3 2.2 1.7]
x ndim:  1
x shape: (3,)
x size:  3
x dtype:  float64

---
```python
# Matrix
x = np.array([[1, 2], [3, 4]])
print("x:\n", x)
print("x ndim: ", x.ndim)
print("x shape:", x.shape)
print("x size: ", x.size)
print("x dtype: ", x.dtype)
```
x:
 [[1 2]
 [3 4]]
x ndim:  2
x shape: (2, 2)
x size:  4
x dtype:  int64

---
```python
# 3-D Tensor
x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("x:\n", x)
print("x ndim: ", x.ndim)
print("x shape:", x.shape)
print("x size: ", x.size)
print("x dtype: ", x.dtype)
```
x:
 [[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
x ndim:  3
x shape: (2, 2, 2)
x size:  8
x dtype:  int64

---
```python
# NumPy also comes with several functions to create tensors quickly.
print("np.zeros((2,2)):\n", np.zeros((2, 2)))
print("np.ones((2,2)):\n", np.ones((2, 2)))
print("np.eye((2)):\n", np.eye(2))                # identity matrix
print("np.random.random((2,2)):\n", np.random.random((2, 2)))
```
np.zeros((2,2)):
 [[0. 0.]
 [0. 0.]]
np.ones((2,2)):
 [[1. 1.]
 [1. 1.]]
np.eye((2)):
 [[1. 0.]
 [0. 1.]]
np.random.random((2,2)):
 [[0.19151945 0.62210877]
 [0.43772774 0.78535858]]

shape)  # brackets are gone
