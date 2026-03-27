from flask import Flask, jsonify, request
import pandas as pd
from datetime import timedelta
import pickle

app = Flask(__name__)

# Load preprocessed data and model
df = pd.read_csv("data/cleaned_sales.csv")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "🔥 E-commerce Sales Forecast API is running 🚀"

@app.route('/predict', methods=['GET'])
def predict():
    # Number of days to forecast
    days = int(request.args.get('days', 7))

    # Last date in dataset
    last_date = pd.to_datetime(df['Order Date'].max())
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]

    # Create features
    X_future = pd.DataFrame({
        "day_of_week": [d.weekday() for d in future_dates],
        "month": [d.month for d in future_dates],
        "year": [d.year for d in future_dates],
        "lag_1": [df['Sales'].iloc[-1]] * days,
        "lag_7": [df['Sales'].iloc[-7]] * days,
    })

    # Predict
    y_pred = model.predict(X_future)

    # Prepare JSON
    result = {str(d.date()): round(float(s), 2) for d, s in zip(future_dates, y_pred)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)