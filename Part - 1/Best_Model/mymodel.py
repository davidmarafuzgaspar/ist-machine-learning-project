import numpy as np
import joblib

# Load the trained pipeline once
model = joblib.load("champion.pkl")

def predict(Xtest):
    # Remove the second sensor (index 1)
    Xtest_mod = np.delete(Xtest, 1, axis=1)
    # Predict with pipeline
    y_pred = model.predict(Xtest_mod)
    
    return y_pred
