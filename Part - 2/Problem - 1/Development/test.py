import pickle
import numpy as np
import joblib


# Load the old pickle
data = joblib.load("Data/Xtrain1.pkl")

# Save as .npy for future use
np.save("Data/Xtrain1.npy", data)

print(data)
print(data.shape)
print("Pickle loaded and saved as .npy successfully!")
