# Part 2 - Problem 1 - Machine Learning Model Development

This directory contains Part 2 - Problem 1 of the Instituto Superior Técnico Machine Learning course project.

# Authors
- David João Marafuz Gaspar - 106541
- Pedro Gaspar Mónico - 106626 

## Project Structure

### 📁 Development/
Contains the complete development pipeline with Jupyter notebooks documenting the entire machine learning process:

- **`0_Features_PreProcessing.ipynb`** - Data preprocessing and feature engineering
- **`1_MLP_Baseline.ipynb`** - Multi-Layer Perceptron baseline model development
- **`2_MLP_Feature_Removal.ipynb`** - MLP with feature removal analysis
- **`3_SVM_Baseline.ipynb`** - Support Vector Machine baseline model
- **`4_SVM_Feature_Removal.ipynb`** - SVM with feature removal analysis
- **`5_Random_Forest.ipynb`** - Random Forest model implementation
- **`6_XGBoost.ipynb`** - XGBoost gradient boosting model
- **`7_Validation.ipynb`** - Model validation and evaluation
- **`8_Champion_Model.ipynb`** - Final champion model selection and training

#### 📁 Data/
Contains the training datasets:
- **`Xtrain1.pkl`** - Training features (pickle file)
- **`Ytrain1.npy`** - Training labels (numpy array)

#### 📁 Validation/
Contains validation results and feature selections:
- **`best_features_mlp.npy`** - Best features selected for MLP model
- **`best_features_svm.npy`** - Best features selected for SVM model

### 🏆 Best_Model/
**This folder contains the final deliverables for the lab assignment:**

- **`mymodel.py`** - **Main prediction function** containing the `predict()` function as required by the lab
- **`champion.pkl`** - **Trained champion model** (pickle file) - the best performing model from all experiments
- **`champion_features.npy`** - **Selected features** used by the champion model
- **`validate_test_code.py`** - Validation script to test the prediction function
- **`Xtrain1.pkl`** - Training features (backup copy)
- **`Ytrain1.npy`** - Training labels (backup copy)

### Key Features of the Champion Model

The champion model implements the following preprocessing and prediction pipeline:

1. **Feature Selection**: Uses pre-selected optimal features for the champion model
2. **Trained Pipeline**: Uses a pre-trained machine learning pipeline loaded from `champion.pkl`
3. **Prediction Function**: Provides a clean `predict(Xtest)` interface that takes test data and returns predictions

## Usage

To use the champion model for predictions:

```python
from mymodel import predict
import numpy as np

# Load your test data
X_test = np.load("your_test_data.npy")  # Shape depends on problem requirements

# Make predictions
predictions = predict(X_test)  # Returns predictions
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
   - Multi-Layer Perceptron (MLP)
   - Support Vector Machines (SVM)
   - Random Forest
   - XGBoost
3. **Feature Selection** - Analyzed feature importance and removal for different models
4. **Model Validation** - Comprehensive validation and performance evaluation
5. **Champion Selection** - Selected the best performing model as the final solution

## Important Notes

- The champion model uses optimized feature selection for optimal performance
- All development work is documented in the Jupyter notebooks in the `Development/` folder
- The final deliverable code is ready for submission in the `Best_Model/` folder
- The `validate_test_code.py` script can be used to verify the prediction function works correctly
- Feature selection results are stored in the `Validation/` folder for reference

---

*This is Part 2 - Problem 1 of the IST Machine Learning course assignment for the academic year 2025–2026.*