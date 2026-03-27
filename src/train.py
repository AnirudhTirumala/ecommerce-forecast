from sklearn.linear_model import LinearRegression


def train_model(data):
    X = data[["lag_1"]]
    y = data["Sales"]
    model = LinearRegression()
    model.fit(X, y)
    return model
