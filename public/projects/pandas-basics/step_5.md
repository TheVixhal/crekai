# Step 5: Merging and Joining DataFrames

Combine multiple DataFrames like SQL joins - essential for working with relational data!

## Your Assignment

### Task 1: Concatenate DataFrames

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Vertical concatenation (stack rows)
result = pd.concat([df1, df2], axis=0)

# Horizontal concatenation (stack columns)
result = pd.concat([df1, df2], axis=1)
```

**Expected output:** Combined DataFrame

---

### Task 2: Merge DataFrames

```python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

salaries = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'salary': [50000, 60000, 55000]
})

# Inner join (SQL-like)
merged = pd.merge(employees, salaries, on='emp_id')

# Left join
merged = pd.merge(employees, salaries, on='emp_id', how='left')
```

**Expected output:** Merged DataFrame with both datasets

---

### Task 3: Join on Index

```python
df1 = pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])
df2 = pd.DataFrame({'B': [3, 4]}, index=['a', 'b'])

# Join on index
result = df1.join(df2)
```

**Expected output:** Joined DataFrame
