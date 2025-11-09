# Part 1 - Machine Learning Model Development

This directory contains Part 1 of the Instituto Superior Técnico Machine Learning course project.
# Authors
- David João Marafuz Gaspar - 106541
- Pedro Gaspar Mónico - 106626 

## Project Structure

### 📁 Development/
Contains the complete development pipeline with Jupyter notebooks documenting the entire machine learning process:

- **`0_Features_Preprocessing.ipynb`** - Data preprocessing and feature engineering
- **`1_Polynomial.ipynb`** - Polynomial regression model development
- **`2_Radial_Basis_Function.ipynb`** - RBF kernel regression experiments
- **`3_Random_Forest.ipynb`** - Random Forest model implementation
- **`4_Decision_Trees.ipynb`** - Decision Tree model development
- **`5_Kernel_Ridge_Regression.ipynb`** - Kernel Ridge Regression experiments
- **`6_Remove_Sensors_KRR.ipynb`** - Sensor removal analysis for KRR
- **`7_XGBoost.ipynb`** - XGBoost gradient boosting model
- **`8_Validation.ipynb`** - Model validation and evaluation
- **`9_Champion_Model.ipynb`** - Final champion model selection and training

#### 📁 Data/
Contains the training datasets:
- **`X_train.npy`** - Training features (numpy array)
- **`Y_train.npy`** - Training labels (numpy array)

### 🏆 Best_Model/
**This folder contains the final deliverables for the lab assignment:**

- **`mymodel.py`** - **Main prediction function** containing the `predict()` function as required by the lab
- **`champion.pkl`** - **Trained champion model** (pickle file) - the best performing model from all experiments
- **`validate_test_code.py`** - Validation script to test the prediction function

### Key Features of the Champion Model

The champion model implements the following preprocessing and prediction pipeline:

1. **Sensor Removal**: Automatically removes the second sensor (index 1) from input data
2. **Trained Pipeline**: Uses a pre-trained machine learning pipeline loaded from `champion.pkl`
3. **Prediction Function**: Provides a clean `predict(Xtest)` interface that takes test data and returns predictions

## Usage

To use the champion model for predictions:

```python
from mymodel import predict
import numpy as np

# Load your test data (should have 6 features)
X_test = np.load("your_test_data.npy")  # Shape: (n_samples, 6)

# Make predictions
predictions = predict(X_test)  # Returns predictions of shape (n_samples,)
```

## Requirements

The project requires the following Python packages:
- numpy
- joblib
- scikit-learn
- Additional packages as used in the development notebooks

## Model Development Process

The development process followed a systematic approach:

1. **Data Preprocessing** - Feature engineering and data cleaning
2. **Multiple Model Testing** - Evaluated various algorithms including:
   - Polynomial Regression
   - Radial Basis Function (RBF) Regression
   - Random Forest
   - Decision Trees
   - Kernel Ridge Regression
   - XGBoost
3. **Feature Selection** - Analyzed sensor importance and removal
4. **Model Validation** - Comprehensive validation and performance evaluation
5. **Champion Selection** - Selected the best performing model as the final solution

## Important Notes

- The champion model automatically handles the removal of the second sensor during prediction
- All development work is documented in the Jupyter notebooks in the `Development/` folder
- The final deliverable code is ready for submission in the `Best_Model/` folder
- The `validate_test_code.py` script can be used to verify the prediction function works correctly

---

*This is Part 1 of the IST Machine Learning course assignment for the academic year 2025–2026.*