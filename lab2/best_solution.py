"""
Best Nonlinear Regression Solution
=================================

Based on analysis of your existing work, this implements the optimal solution
combining the best performing models with proper preprocessing.

Key findings from your analysis:
- Kernel Ridge RBF: R² = 0.9926 (best individual)
- SVR RBF: R² = 0.9899
- RBF + Ridge: R² = 0.9698
- Spline: R² = 0.9480
- Polynomial: R² = 0.9167

Optimal solution: Ensemble of best models with advanced preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess data with outlier removal and scaling."""
    print("Loading and preprocessing data...")
    
    # Load data
    X = np.load("Data/X_train.npy")
    y = np.load("Data/Y_train.npy")
    print(f"Original data shape: X={X.shape}, y={y.shape}")
    
    # Outlier detection and removal
    X_scaled = StandardScaler().fit_transform(X)
    
    # Z-score based outlier detection
    zs = np.abs(zscore(X_scaled))
    z_thresh = 3.0
    outlier_mask_z = (zs > z_thresh).any(axis=1)
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    is_outlier_iso = iso.fit_predict(X_scaled) == -1
    
    # Combine methods
    outlier_union = outlier_mask_z | is_outlier_iso
    clean_mask = ~outlier_union
    
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]
    
    print(f"Outliers removed: {outlier_union.sum()}")
    print(f"Clean data shape: {X_clean.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # PCA (keep 99% variance)
    pca = PCA(n_components=0.99)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA components: {X_pca.shape[1]}")
    
    return X_pca, y_clean, scaler, pca

def create_best_models():
    """Create the best performing models based on your analysis."""
    models = {
        'Kernel Ridge RBF (Best)': KernelRidge(
            kernel='rbf', 
            gamma=0.12, 
            alpha=0.0022
        ),
        'Kernel Ridge RBF (Alt)': KernelRidge(
            kernel='rbf', 
            gamma=0.20, 
            alpha=0.0024
        ),
        'SVR RBF': SVR(
            kernel='rbf',
            gamma=0.1,
            C=1000,
            epsilon=0.01
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
    }
    return models

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Evaluate all models and return results."""
    print("\nEvaluating individual models...")
    print("=" * 50)
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        }
        
        print(f"{name}:")
        print(f"  Train R²: {train_r2:.6f}")
        print(f"  Test R²:  {test_r2:.6f}")
        print(f"  Test RMSE: {test_rmse:.6f}")
        print()
    
    return results

def create_ensemble(results):
    """Create ensemble from best models."""
    print("Creating optimal ensemble...")
    
    # Select best models for ensemble
    ensemble_models = [
        ('kernel_ridge_best', results['Kernel Ridge RBF (Best)']['model']),
        ('kernel_ridge_alt', results['Kernel Ridge RBF (Alt)']['model']),
        ('svr_rbf', results['SVR RBF']['model']),
        ('random_forest', results['Random Forest']['model'])
    ]
    
    # Weighted ensemble based on individual performance
    weights = [0.35, 0.25, 0.25, 0.15]
    
    ensemble = VotingRegressor(
        estimators=ensemble_models,
        weights=weights
    )
    
    return ensemble

def cross_validate_models(models, X, y):
    """Perform cross-validation on all models."""
    print("\nCross-validation analysis...")
    print("=" * 50)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        cv_results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        print(f"{name}: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    
    return cv_results

def plot_results(y_test, y_pred, model_name, r2_score_val):
    """Plot prediction results."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - R² = {r2_score_val:.6f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main function implementing the optimal solution."""
    print("=" * 60)
    print("BEST NONLINEAR REGRESSION SOLUTION")
    print("=" * 60)
    
    # Load and preprocess data
    X_processed, y_processed, scaler, pca = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=200, random_state=42, shuffle=True
    )
    
    print(f"\nData split: Train={X_train.shape}, Test={X_test.shape}")
    
    # Create models
    models = create_best_models()
    
    # Evaluate individual models
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    
    # Cross-validation
    cv_results = cross_validate_models(models, X_train, y_train)
    
    # Create and evaluate ensemble
    ensemble = create_ensemble(results)
    ensemble.fit(X_train, y_train)
    
    y_train_ensemble = ensemble.predict(X_train)
    y_test_ensemble = ensemble.predict(X_test)
    
    ensemble_train_r2 = r2_score(y_train, y_train_ensemble)
    ensemble_test_r2 = r2_score(y_test, y_test_ensemble)
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_ensemble))
    
    print("OPTIMAL ENSEMBLE MODEL")
    print("=" * 50)
    print(f"Train R²: {ensemble_train_r2:.6f}")
    print(f"Test R²:  {ensemble_test_r2:.6f}")
    print(f"Test RMSE: {ensemble_test_rmse:.6f}")
    
    # Plot best individual model vs ensemble
    best_individual = max(results.items(), key=lambda x: x[1]['test_r2'])
    best_name, best_result = best_individual
    
    print(f"\nBest individual model: {best_name} (R² = {best_result['test_r2']:.6f})")
    print(f"Ensemble model: R² = {ensemble_test_r2:.6f}")
    
    # Plot results
    plot_results(y_test, best_result['model'].predict(X_test), 
                f"Best Individual ({best_name})", best_result['test_r2'])
    plot_results(y_test, y_test_ensemble, "Optimal Ensemble", ensemble_test_r2)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    print("✅ BEST APPROACH: Ensemble of top-performing models")
    print(f"✅ FINAL R² SCORE: {ensemble_test_r2:.6f}")
    print("✅ COMPONENTS:")
    print("   • Kernel Ridge RBF (gamma=0.12, alpha=0.0022)")
    print("   • Kernel Ridge RBF (gamma=0.20, alpha=0.0024)")
    print("   • SVR RBF (gamma=0.1, C=1000)")
    print("   • Random Forest (200 trees, max_depth=15)")
    print("✅ PREPROCESSING:")
    print("   • Outlier removal (Z-score + Isolation Forest)")
    print("   • Feature scaling (StandardScaler)")
    print("   • Dimensionality reduction (PCA, 99% variance)")
    print("✅ WEIGHTED ENSEMBLE: [0.35, 0.25, 0.25, 0.15]")
    
    return ensemble, results, ensemble_test_r2

if __name__ == "__main__":
    model, individual_results, final_r2 = main()