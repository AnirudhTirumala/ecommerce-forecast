# train.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load preprocessed data
data_path = "data/preprocessed_data.csv"  # Update with your actual file path
data = pd.read_csv(data_path)

# Assume last column is target, rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set and print error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model training complete. Test MSE: {mse}")

# Ensure the models folder exists
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/model.pkl")
print("Model saved at models/model.pkl")
