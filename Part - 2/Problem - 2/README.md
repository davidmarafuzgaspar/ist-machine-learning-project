# Part 2 - Problem 2 - Machine Learning Model Development

This directory contains Part 2 - Problem 2 of the Instituto Superior Técnico Machine Learning course project.

# Authors
- David João Marafuz Gaspar - 106541
- Pedro Gaspar Mónico - 106626 

## Project Structure

### 📁 Development/
Contains the complete development pipeline with Jupyter notebooks documenting the entire machine learning process:

- **`0_Features_PreProcessing.ipynb`** - Data preprocessing and initial feature engineering
- **`1_Baseline_V1_Features.ipynb`** - Baseline model with V1 feature set
- **`2_Feature_V1_Engineering.ipynb`** - Feature engineering and optimization for V1 features
- **`3_Baseline_V2_Features.ipynb`** - Baseline model with V2 feature set
- **`4_Feature_V2_Engineering.ipynb`** - Feature engineering and optimization for V2 features
- **`5_SVM_Champion.ipynb`** - Support Vector Machine champion model development
- **`6_Random_Forest_Champion.ipynb`** - Random Forest champion model development
- **`7_Champion_Model.ipynb`** - Final champion model selection and training

#### 📁 Data/
Contains the training datasets:
- **`Xtrain2.pkl`** - Training features (pickle file with skeleton sequences)
- **`Ytrain2.npy`** - Training labels (numpy array)
- **`feature_importance_v1_baseline.csv`** - Feature importance analysis for V1 baseline
- **`feature_importance_v2_baseline.csv`** - Feature importance analysis for V2 baseline

### 🏆 Best_Model/
**This folder contains the final deliverables for the lab assignment:**

- **`mymodel.py`** - **Main prediction function** containing the `predict()` function as required by the lab
- **`feature_extraction.py`** - **Feature extraction module** for extracting features from skeleton sequences
- **`champion.pkl`** - **Trained champion model** (pickle file) - the best performing model from all experiments
- **`validate_test_code.py`** - Validation script to test the prediction function
- **`Xtrain2.pkl`** - Training features (backup copy)
- **`Ytrain2.npy`** - Training labels (backup copy)

### Key Features of the Champion Model

The champion model implements the following preprocessing and prediction pipeline:

1. **Feature Extraction**: Extracts combined features from skeleton sequences using velocity, acceleration, and asymmetry metrics
2. **Feature Masking**: Uses pre-selected optimal features via feature masking for the champion model
3. **Exercise Encoding**: Encodes exercise types to incorporate exercise-specific information
4. **Trained Pipeline**: Uses a pre-trained machine learning pipeline loaded from `champion.pkl`
5. **Patient-Level Aggregation**: Aggregates sequence-level predictions to patient-level predictions using enhanced majority voting
6. **Prediction Function**: Provides a clean `predict(Xtest)` interface that takes test data and returns patient-level predictions

## Usage

To use the champion model for predictions:

```python
from mymodel import predict
import pandas as pd

# Load your test data
# X_test should be a DataFrame with columns: ['Patient_Id', 'Exercise_Id', 'Skeleton_Sequence']
X_test = pd.read_pickle("your_test_data.pkl")

# Make predictions
predictions = predict(X_test)  # Returns patient-level predictions (0 for left, 1 for right)
```

## Requirements

The project requires the following Python packages:
- numpy
- pandas
- joblib
- scikit-learn
- Additional packages as used in the development notebooks

## Model Development Process

The development process followed a systematic approach:

1. **Data Preprocessing** - Feature engineering from skeleton sequences including:
   - Velocity and acceleration calculations
   - Asymmetry metrics between left and right body sides
   - Statistical aggregations (mean, std, max, min)
2. **Feature Set Development** - Developed and evaluated two feature sets:
   - **V1 Features**: Initial feature set with basic statistics
   - **V2 Features**: Enhanced feature set with additional engineered features
3. **Multiple Model Testing** - Evaluated various algorithms including:
   - Support Vector Machines (SVM)
   - Random Forest
4. **Feature Selection** - Analyzed feature importance and applied feature masking
5. **Model Validation** - Comprehensive validation and performance evaluation
6. **Champion Selection** - Selected the best performing model as the final solution
7. **Patient-Level Aggregation** - Implemented enhanced majority voting to aggregate sequence-level predictions to patient-level predictions

## Important Notes

- The champion model uses optimized feature selection via feature masking for optimal performance
- All development work is documented in the Jupyter notebooks in the `Development/` folder
- The final deliverable code is ready for submission in the `Best_Model/` folder
- The `validate_test_code.py` script can be used to verify the prediction function works correctly
- Feature importance analyses are stored as CSV files in the `Development/` folder for reference
- The model predicts the impaired side (0 for left, 1 for right) at the patient level
- Input data must be a DataFrame with columns: `['Patient_Id', 'Exercise_Id', 'Skeleton_Sequence']`

---

*This is Part 2 - Problem 3 of the IST Machine Learning course assignment for the academic year 2025–2026.*
