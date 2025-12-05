# IST Machine Learning Project

This repository contains the complete project for the Instituto Superior Técnico Machine Learning course.

## Authors
- David João Marafuz Gaspar - 106541
- Pedro Gaspar Mónico - 106626

## Project Overview

This project consists of two main parts, each addressing different machine learning challenges:

- **Part 1**: Regression problem with sensor data
- **Part 2**: Two classification problems with different data types and feature engineering approaches

## Repository Structure

```
ist-machine-learning-project/
├── Part - 1/                    # Regression problem
│   ├── Best_Model/             # Final deliverable model
│   ├── Development/            # Development notebooks
│   └── README.md               # Part 1 documentation
│
├── Part - 2/                   # Classification problems
│   ├── Papers/                 # Reference papers
│   ├── Problem - 1/            # Classification problem 1
│   │   ├── Best_Model/         # Final deliverable model
│   │   ├── Development/        # Development notebooks
│   │   └── README.md           # Problem 1 documentation
│   └── Problem - 2/            # Classification problem 2
│       ├── Best_Model/         # Final deliverable model
│       ├── Development/        # Development notebooks
│       └── README.md           # Problem 2 documentation
│
├── Project.pdf                 # Project specification
└── Report.pdf                  # Project report
```

## Part 1 - Regression Problem

**Location**: `Part - 1/`

A regression problem involving sensor data where the goal is to predict continuous values. The champion model uses Kernel Ridge Regression with sensor removal optimization.

**Key Features**:
- Multiple regression algorithms tested (Polynomial, RBF, Random Forest, Decision Trees, KRR, XGBoost)
- Sensor removal analysis for feature optimization
- Final model: Kernel Ridge Regression with optimized sensor configuration

**See**: [Part 1 README](Part%20-%201/README.md) for detailed documentation.

## Part 2 - Classification Problems

### Problem 1 - Feature Selection Classification

**Location**: `Part - 2/Problem - 1/`

A classification problem focusing on feature selection and model comparison. The champion model uses optimized feature selection with multiple algorithm evaluations.

**Key Features**:
- Multiple classification algorithms (MLP, SVM, Random Forest, XGBoost)
- Feature removal analysis for different models
- Final model with pre-selected optimal features

**See**: [Part 2 - Problem 1 README](Part%20-%202/Problem%20-%201/README.md) for detailed documentation.

### Problem 2 - Skeleton Sequence Classification

**Location**: `Part - 2/Problem - 2/`

A classification problem involving skeleton sequence data where the goal is to predict the impaired side of patients. The champion model extracts features from skeleton sequences and aggregates predictions to the patient level.

**Key Features**:
- Feature extraction from skeleton sequences (velocity, acceleration, asymmetry metrics)
- Two feature sets developed (V1 and V2)
- Patient-level aggregation using enhanced majority voting
- Multiple algorithms tested (SVM, Random Forest)

**See**: [Part 2 - Problem 2 README](Part%20-%202/Problem%20-%202/README.md) for detailed documentation.

## Requirements

The project requires the following Python packages:
- numpy
- pandas
- joblib
- scikit-learn
- xgboost
- Additional packages as specified in individual part READMEs

## Development Process

Each part follows a systematic development approach:

1. **Data Preprocessing** - Feature engineering and data cleaning
2. **Model Exploration** - Testing multiple algorithms and configurations
3. **Feature Engineering** - Optimizing features for each problem
4. **Model Validation** - Comprehensive validation and performance evaluation
5. **Champion Selection** - Selecting the best performing model

All development work is documented in Jupyter notebooks within each part's `Development/` directory.

## Deliverables

Each part contains a `Best_Model/` directory with:
- `mymodel.py` - Main prediction function
- `champion.pkl` - Trained champion model
- `validate_test_code.py` - Validation script
- Additional files as needed (feature extraction, feature selections, etc.)

## Documentation

- [Part 1 Documentation](Part%20-%201/README.md)
- [Part 2 - Problem 1 Documentation](Part%20-%202/Problem%20-%201/README.md)
- [Part 2 - Problem 2 Documentation](Part%20-%202/Problem%20-%202/README.md)

## Important Notes

- All development work is documented in Jupyter notebooks
- Final deliverable code is ready for submission in each `Best_Model/` directory
- Validation scripts are provided to test the prediction functions
- Each part is independent and can be used separately

---

*This is a IST Machine Learning course assignment for the academic year 2025–2026.*
