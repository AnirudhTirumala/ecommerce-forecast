import pandas as pd


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df["lag_1"] = df["Sales"].shift(1).fillna(0)
    return df
    
