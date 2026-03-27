import pandas as pd

# Load CSV
df = pd.read_csv("data/train.csv")

# Convert 'Order Date' to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Create day_of_week column
df["day_of_week"] = df["Order Date"].dt.weekday

# Create lag feature
df["lag_1"] = df["Sales"].shift(1).bfill()

# Save processed data
df.to_csv("data/processed.csv", index=False)
# Last line of code
return pred

# <-- Make sure there's a blank line here (just press Enter)