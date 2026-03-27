from src.preprocess import preprocess_data
from src.train import train_model
from src.predict import predict_model


def main():
    data_file = "data/train.csv"
    data = preprocess_data(data_file)
    model = train_model(data)
    predictions = predict_model(model, data)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
