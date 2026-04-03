def predict_model(model, data):
    X = data[["lag_1"]]
    predictions = model.predict(X)
    return predictions
    
