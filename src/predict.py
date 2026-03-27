import pandas as pd
import pickle
from datetime import datetime, timedelta

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare input for next 7 days
last_date = datetime.today()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

X_new = pd.DataFrame({
    "day_of_week": [d.weekday() for d in future_dates],
    "month": [d.month for d in future_dates],
    "year": [d.year for d in future_dates],
    "lag_1": [180.6] * 7,
    "lag_7": [180.5] * 7
})

preds = model.predict(X_new)

print("Next 7 Days Sales Forecast:\n")
for d, p in zip(future_dates, preds):
    print(f"{d.strftime('%Y-%m-%d')} -> {round(p,2)}")