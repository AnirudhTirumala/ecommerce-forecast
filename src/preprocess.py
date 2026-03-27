import pandas as pd


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["day_of_week"] = df["Order Date"].dt.weekday  # 0=Mon, 6=Sun

    # Create lag feature
    df["lag_1"] = df["Sales"].shift(1).bfill()

    return df

