import numpy as np
import joblib
from sklearn.metrics import f1_score
from mymodel import predict  

# Load test data
X_test = joblib.load("Xtrain1.pkl")  # DataFrame with Patient_Id and Skeleton_Features
y_test = np.load("Ytrain1.npy")

# Make predictions
y_pred = predict(X_test)

# Validate shape
if y_pred.shape != y_test.shape:
    raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_test.shape}")  
print("Prediction format is valid.")

# Compute F1-score (macro, for multi-class)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-score on test data: {f1:.4f}")
