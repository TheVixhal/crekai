# Step 6: Time Series and Advanced Topics

Explore time series data and advanced Pandas features - prepare for real-world data analysis!

## What You'll Learn

- Working with dates and times
- Time series operations
- Resampling and rolling windows
- Advanced techniques

---

## Time Series Basics

### Creating Date Ranges

```python
import pandas as pd

# Date range
dates = pd.date_range('2024-01-01', periods=10, freq='D')

# DataFrame with dates
df = pd.DataFrame({
    'date': dates,
    'value': range(10)
})

# Set date as index
df = df.set_index('date')
```

---

## Working with Dates

### Parsing Dates

```python
# Read CSV with date parsing
df = pd.read_csv('data.csv', parse_dates=['date_column'])

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()
```

---

## Resampling

```python
# Daily data to monthly
monthly = df.resample('M').mean()

# Hourly to daily
daily = df.resample('D').sum()
```

---

## Rolling Windows

```python
# 7-day rolling average
df['rolling_avg'] = df['value'].rolling(window=7).mean()

# Expanding window
df['cumsum'] = df['value'].expanding().sum()
```

---

## Advanced Topics Preview

- Multi-index DataFrames
- Categorical data
- Custom aggregations
- Performance optimization

---

## Congratulations! 🎉

You've mastered Pandas fundamentals:
- ✅ DataFrames and Series
- ✅ Data selection and filtering
- ✅ Data cleaning
- ✅ GroupBy aggregations
- ✅ Merging DataFrames
- ✅ Time series basics

**You're ready for real data analysis projects!**

---

## What's Next?

Apply your skills in:
- **Build ANN** - Prepare datasets for neural networks
- **Machine learning projects** - Feature engineering
- **Data analysis** - Exploratory data analysis

Continue learning! 🚀
