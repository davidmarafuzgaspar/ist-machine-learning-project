import numpy as np
import joblib
from sklearn.metrics import balanced_accuracy_score
from mymodel import predict  

# Load test data
X_test = joblib.load("Xtrain2.pkl")  # DataFrame with Patient_Id, Exercise_Id, and  Skeleton_Sequence
y_test = np.load("Ytrain2.npy")

# Make predictions
y_pred = predict(X_test)

# Validate shape
if y_pred.shape != y_test.shape:
    raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_test.shape}")  
print("Prediction format is valid.")

# Compute F1-score (macro, for multi-class)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy on test data: {balanced_accuracy:.4f}")
