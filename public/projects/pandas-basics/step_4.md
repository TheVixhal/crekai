# Step 4: GroupBy and Aggregations

Learn to summarize and analyze data by groups - essential for data analysis!

## Your Assignment

### Task 1: GroupBy Basics

```python
df = pd.DataFrame({
    'Department': ['Sales', 'Sales', 'IT', 'IT', 'HR'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Salary': [50000, 55000, 60000, 58000, 52000]
})

# Group by department
grouped = df.groupby('Department')

# Aggregations
mean_salary = grouped['Salary'].mean()
total_salary = grouped['Salary'].sum()
count = grouped.size()
```

**Expected output:** Aggregated results by group

---

### Task 2: Multiple Aggregations

```python
# Multiple statistics at once
stats = df.groupby('Department')['Salary'].agg(['mean', 'sum', 'count', 'std'])

# Custom aggregations
stats = df.groupby('Department').agg({
    'Salary': ['mean', 'max', 'min'],
    'Employee': 'count'
})
```

**Expected output:** DataFrame with multiple statistics

---

### Task 3: Pivot Tables

```python
# Create pivot table
pivot = df.pivot_table(
    values='Salary',
    index='Department',
    aggfunc='mean'
)
```

**Expected output:** Summarized pivot table
