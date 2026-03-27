import pandas as pd
from src.preprocess import preprocess_data


def load_test_data(file_path):
    df = preprocess_data(file_path)
    return df
