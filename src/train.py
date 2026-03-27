import pandas as pd
import pickle
import os

data_path = "data/processed.csv"
df = pd.read_csv(data_path)

# Simple model: sum sales per day of week
model = df.groupby("day_of_week")["Sales"].mean().to_dict()

# Save model
model_path = os.path.join("models", "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)