from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return "E-commerce Sales Forecast API 🚀"


@app.route("/predict", methods=["GET"])
def predict():
    # Get number of days from query parameter, default 7
    days = int(request.args.get("days", 7))
    last_date = datetime.today()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]

    X_new = pd.DataFrame({
        "day_of_week": [d.weekday() for d in future_dates],
        "month": [d.month for d in future_dates],
        "year": [d.year for d in future_dates],
        "lag_1": [180.6] * days,  # Example placeholder
        "lag_7": [180.5] * days
    })

    preds = model.predict(X_new)
    result = {d.strftime("%Y-%m-%d"): round(p, 2) for d, p in zip(future_dates, preds)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)