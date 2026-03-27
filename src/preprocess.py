import pandas as pd

# Load sales data
df = pd.read_csv("data/sales.csv")

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Feature engineering
df['day_of_week'] = df['Order Date'].dt.weekday  # 0=Mon, 6=Sun
df['month'] = df['Order Date'].dt.month
df['year'] = df['Order Date'].dt.year

# Lag features
df['lag_1'] = df['Sales'].shift(1).bfill()
df['lag_7'] = df['Sales'].shift(7).bfill()

# Save processed data
df.to_csv("data/sales_processed.csv", index=False)
print(df.head())