import pandas as pd
from src.preprocess import preprocess_data
from sklearn.linear_model import LinearRegression
import pickle


def train_model(file_path):
    df = preprocess_data(file_path)
    X = pd.get_dummies(df["day_of_week"])
    y = df["Sales"]

    model = LinearRegression()
    model.fit(X, y)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model
