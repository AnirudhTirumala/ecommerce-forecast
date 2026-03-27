import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load processed data
df = pd.read_csv("data/sales_processed.csv")

# Features and target
X = df[['day_of_week', 'month', 'year', 'lag_1', 'lag_7']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to models/model.pkl")