import pandas as pd

# Load raw data
df = pd.read_csv("data/sales.csv")

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Order Date'])

# Keep only needed columns
df = df[['Order Date', 'Sales']].sort_values('Order Date')

# Feature engineering
df['day_of_week'] = df['Order Date'].dt.weekday  # 0=Mon, 6=Sun
df['month'] = df['Order Date'].dt.month
df['year'] = df['Order Date'].dt.year

# Lag features (previous day and previous week sales)
df['lag_1'] = df['Sales'].shift(1).bfill()
df['lag_7'] = df['Sales'].shift(7).bfill()

# Save cleaned data
df.to_csv("data/cleaned_sales.csv", index=False)

# Print sample
print(df.head())