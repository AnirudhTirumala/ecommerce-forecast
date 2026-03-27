import pandas as pd
import pickle


def make_forecast(future_dates):
    # Load trained model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df_future = pd.DataFrame({"date": future_dates})
    df_future["day_of_week"] = df_future["date"].apply(lambda x: x.weekday())

    predictions = model.predict(df_future)
    return predictions.tolist()

