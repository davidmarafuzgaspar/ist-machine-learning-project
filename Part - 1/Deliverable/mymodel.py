import numpy as np
import joblib

# Load the trained pipeline once
model = joblib.load("champion.pkl")

def predict(Xtest):
    # Remove the first sensor (index 0) -> same as training
    Xtest_mod = np.delete(Xtest, 0, axis=1)
    # Predict with pipeline
    y_pred = model.predict(Xtest_mod)
    
    return y_pred
