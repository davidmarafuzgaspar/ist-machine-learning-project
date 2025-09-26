# Optimal Nonlinear Regression Solution
#
# This script provides a comprehensive solution for nonlinear regression,
# combining advanced techniques with proper validation and analysis.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ----------------------------------------------------------------------
# 1. Data Loading and Initial Analysis
# ----------------------------------------------------------------------

# Load data
X = np.load("Data/X_train.npy")  # shape (700, 6)
y = np.load("Data/Y_train.npy")  # shape (700,)

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Target statistics: mean={y.mean():.3f}, std={y.std():.3f}")
print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

# Data visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(X.shape[1]):
    axes[i].scatter(X[:, i], y, alpha=0.6, s=20)
    axes[i].set_xlabel(f'Feature {i+1}')
    axes[i].set_ylabel('Target')
    axes[i].set_title(f'Feature {i+1} vs Target')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 2. Advanced Data Preprocessing
# ----------------------------------------------------------------------

def advanced_outlier_detection(X, y, contamination=0.05):
    """Advanced outlier detection combining multiple methods"""
    
    # Z-score method
    X_scaled = StandardScaler().fit_transform(X)
    z_scores = np.abs(zscore(X_scaled))
    z_outliers = (z_scores > 3).any(axis=1)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_outliers = iso_forest.fit_predict(X_scaled) == -1
    
    # Target-based outlier detection
    y_scaled = (y - y.mean()) / y.std()
    y_outliers = np.abs(y_scaled) > 3
    
    # Combine all methods
    combined_outliers = z_outliers | iso_outliers | y_outliers
    
    print(f"Outliers detected: {combined_outliers.sum()} ({combined_outliers.mean()*100:.1f}%)")
    print(f"Z-score outliers: {z_outliers.sum()}")
    print(f"Isolation Forest outliers: {iso_outliers.sum()}")
    print(f"Target outliers: {y_outliers.sum()}")
    
    return ~combined_outliers  # Return mask for clean data

# Apply outlier detection
clean_mask = advanced_outlier_detection(X, y)
X_clean = X[clean_mask]
y_clean = y[clean_mask]

print(f"Clean data shape: {X_clean.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.25, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ----------------------------------------------------------------------
# 3. Feature Engineering and Scaling
# ----------------------------------------------------------------------

def create_feature_engineering_pipeline():
    """Create comprehensive feature engineering pipeline"""
    
    # Multiple scaling options
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler()
    }
    
    # PCA options
    pca_options = {
        'no_pca': None,
        'pca_95': PCA(n_components=0.95),
        'pca_99': PCA(n_components=0.99)
    }
    
    return scalers, pca_options

scalers, pca_options = create_feature_engineering_pipeline()

# ----------------------------------------------------------------------
# 4. Advanced RBF Implementation
# ----------------------------------------------------------------------

def advanced_rbf_transform(X, centers, sigma, method='gaussian'):
    """Advanced RBF transformation with multiple kernel types"""
    
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    if method == 'gaussian':
        return np.exp(-np.sum(diff**2, axis=2) / (2*sigma**2))
    elif method == 'multiquadric':
        return np.sqrt(np.sum(diff**2, axis=2) + sigma**2)
    elif method == 'inverse_multiquadric':
        return 1.0 / np.sqrt(np.sum(diff**2, axis=2) + sigma**2)
    elif method == 'thin_plate_spline':
        r = np.sqrt(np.sum(diff**2, axis=2))
        return r**2 * np.log(r + 1e-10)  # avoid log(0)
    else:
        raise ValueError(f"Unknown RBF method: {method}")

def optimize_rbf_centers(X, n_centers, method='kmeans'):
    """Optimize RBF center selection"""
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
        kmeans.fit(X)
        return kmeans.cluster_centers_
    elif method == 'random':
        np.random.seed(42)
        return X[np.random.choice(X.shape[0], n_centers, replace=False)]
    elif method == 'uniform':
        centers = np.zeros((n_centers, X.shape[1]))
        for i in range(X.shape[1]):
            centers[:, i] = np.linspace(X[:, i].min(), X[:, i].max(), n_centers)
        return centers
    else:
        raise ValueError(f"Unknown center method: {method}")

# ----------------------------------------------------------------------
# 5. Comprehensive Model Suite
# ----------------------------------------------------------------------

def get_model_suite():
    """Get comprehensive suite of nonlinear regression models"""
    
    models = {
        # Kernel methods
        'KernelRidge_RBF': KernelRidge(kernel='rbf'),
        'KernelRidge_Poly': KernelRidge(kernel='polynomial'),
        'KernelRidge_Linear': KernelRidge(kernel='linear'),
        'SVR_RBF': SVR(kernel='rbf'),
        'SVR_Poly': SVR(kernel='poly'),
        
        # Tree-based methods
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        
        # Neural networks
        'MLP_Default': MLPRegressor(random_state=42, max_iter=1000),
        'MLP_Large': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
        
        # Linear with regularization
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(random_state=42)
    }
    
    return models

def get_hyperparameter_grids():
    """Get hyperparameter grids for each model"""
    
    param_grids = {
        'KernelRidge_RBF': {'alpha': [0.001, 0.01, 0.1, 1.0], 'gamma': [0.01, 0.1, 1.0, 10.0]},
        'KernelRidge_Poly': {'alpha': [0.001, 0.01, 0.1, 1.0], 'degree': [2, 3, 4, 5]},
        'SVR_RBF': {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1.0], 'epsilon': [0.01, 0.1, 0.2]},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
        'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
        'ElasticNet': {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}
    }
    
    return param_grids

# ----------------------------------------------------------------------
# 6. Advanced RBF with Multiple Kernels
# ----------------------------------------------------------------------

def optimize_rbf_model(X_train, y_train, X_test, y_test):
    """Optimize RBF model with multiple kernels and center selection methods"""
    
    sigma_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_centers_list = [50, 100, 150, 200, 250]
    alpha_list = [0.001, 0.01, 0.1, 1.0]
    rbf_methods = ['gaussian', 'multiquadric', 'inverse_multiquadric']
    center_methods = ['kmeans', 'random', 'uniform']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -np.inf
    best_params = None
    results = []
    
    print("Optimizing RBF model...")
    
    for sigma, n_centers, alpha, rbf_method, center_method in product(
        sigma_list, n_centers_list, alpha_list, rbf_methods, center_methods
    ):
        scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            try:
                centers = optimize_rbf_centers(X_tr, n_centers, center_method)
                X_tr_rbf = advanced_rbf_transform(X_tr, centers, sigma, rbf_method)
                X_val_rbf = advanced_rbf_transform(X_val, centers, sigma, rbf_method)
                
                model = Ridge(alpha=alpha)
                model.fit(X_tr_rbf, y_tr)
                
                y_val_pred = model.predict(X_val_rbf)
                scores.append(r2_score(y_val, y_val_pred))
            except:
                scores.append(-np.inf)
        
        mean_score = np.mean(scores)
        results.append({
            'sigma': sigma, 'n_centers': n_centers, 'alpha': alpha,
            'rbf_method': rbf_method, 'center_method': center_method,
            'score': mean_score
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                'sigma': sigma, 'n_centers': n_centers, 'alpha': alpha,
                'rbf_method': rbf_method, 'center_method': center_method
            }
    
    print("Best RBF model score:", best_score)
    print("Best RBF model parameters:", best_params)
    return best_params, results
