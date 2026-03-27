import pickle
import os
from datetime import datetime

model_path = os.path.join(os.path.dirname(__file__), "../models/model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)


def predict_sales(date: datetime) -> float:
    """
    Predict sales for a given date.

    Args:
        date (datetime): The date to predict sales for.

    Returns:
        float: Predicted sales value.
    """
    day_of_week = date.weekday()
    # Example: simple prediction using model (replace with actual logic)
    pred = model.get(day_of_week, 0.0)
    return pred