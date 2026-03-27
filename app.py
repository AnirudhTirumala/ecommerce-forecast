from flask import Flask, request, jsonify
from src.predict import predict_sales
from datetime import datetime, timedelta

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    start_date_str = data.get("start_date")
    days = int(data.get("days", 7))

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    future_dates = [start_date + timedelta(days=i) for i in range(days)]

    response = []
    for date in future_dates:
        pred = predict_sales(date)
        response.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "predicted_sales": float(pred),
                "day_of_week": date.weekday(),
            }
        )

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
