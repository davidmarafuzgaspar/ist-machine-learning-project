import numpy as np
import joblib

# Load the trained pipeline
model = joblib.load("champion.pkl") 

# Load the selected features used during training
selected_features = np.load("champion_features.npy")

def predict(Xtest):
    # Convert Skeleton_Features column into a 2D NumPy array
    Xtest_array = np.vstack(Xtest["Skeleton_Features"].values)
    # Keep only the selected features
    Xtest_mod = Xtest_array[:, selected_features]
    # Predict using the pipeline
    y_pred = model.predict(Xtest_mod)
    
    return y_pred
