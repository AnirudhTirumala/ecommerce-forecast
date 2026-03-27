from flask import Flask, request, jsonify
from src.predict import make_forecast
from datetime import datetime, timedelta

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    days = data.get("days", 7)
    last_date = datetime.today()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    predictions = make_forecast(future_dates)

    response = {
        "predictions": predictions,
        "day_of_week": [d.weekday() for d in future_dates]
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
