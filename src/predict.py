import pandas as pd
from datetime import timedelta
import pickle

# Load data and model
df = pd.read_csv("data/cleaned_sales.csv")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Next 7 days
last_date = pd.to_datetime(df['Order Date'].max())
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

# Features for prediction
X_future = pd.DataFrame({
    "day_of_week": [d.weekday() for d in future_dates],
    "month": [d.month for d in future_dates],
    "year": [d.year for d in future_dates],
    "lag_1": [df['Sales'].iloc[-1]] * 7,
    "lag_7": [df['Sales'].iloc[-7]] * 7,
})

# Predict
y_pred = model.predict(X_future)

# Display
print("Next 7 Days Sales Forecast:\n")
for d, s in zip(future_dates, y_pred):
    print(f"{d.date()} -> {round(s, 2)}")