"""
Simple Best Solution - Just Scale Data
=====================================

Uses only StandardScaler and your best Kernel Ridge RBF model.
No PCA, no outlier removal, no ensemble - just the optimal model with scaling.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    """Simple solution with just scaling and best model."""
    print("=" * 50)
    print("SIMPLE BEST SOLUTION - JUST SCALE DATA")
    print("=" * 50)
    
    # Load data
    X = np.load("Data/X_train.npy")
    y = np.load("Data/Y_train.npy")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=200, random_state=42, shuffle=True
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # ONLY scale the data - nothing else
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data scaled with StandardScaler")
    
    # Your best model from analysis
    model = KernelRidge(kernel='rbf', gamma=0.12, alpha=0.0022)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    print("Model trained")
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Train R²: {train_r2:.6f}")
    print(f"Test R²:  {test_r2:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Kernel Ridge RBF - R² = {test_r2:.6f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 50)
    print("SOLUTION SUMMARY")
    print("=" * 50)
    print("✅ Model: Kernel Ridge RBF")
    print("✅ Parameters: gamma=0.12, alpha=0.0022")
    print("✅ Preprocessing: StandardScaler only")
    print(f"✅ Performance: R² = {test_r2:.6f}")
    print("✅ Simple and effective!")
    
    return model, scaler, test_r2

if __name__ == "__main__":
    model, scaler, r2 = main()