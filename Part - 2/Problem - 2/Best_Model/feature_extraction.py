import numpy as np
import pandas as pd


# Define left and right side keypoints
LEFT_SIDE_KEYPOINTS = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]  # Left body parts
RIGHT_SIDE_KEYPOINTS = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # Right body parts

def calculate_velocity(sequence):
    """Calculate velocity between consecutive frames"""
    # sequence shape: (seq_length, 33, 2)
    velocity = np.diff(sequence, axis=0)  # (seq_length-1, 33, 2)
    return velocity

def calculate_side_velocity_stats(velocity_sequence, side_keypoints):
    """Calculate velocity statistics for a body side"""
    # velocity_sequence shape: (seq_length-1, 33, 2)
    side_velocity = velocity_sequence[:, side_keypoints, :]  # (seq_length-1, n_keypoints, 2)
    
    # Calculate magnitude of velocity vectors
    velocity_magnitudes = np.sqrt(np.sum(side_velocity**2, axis=2))  # (seq_length-1, n_keypoints)
    
    # Aggregate statistics
    mean_velocity = np.mean(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)
    max_velocity = np.max(velocity_magnitudes)
    
    return mean_velocity, std_velocity, max_velocity

def calculate_acceleration(velocity_sequence):
    """Calculate acceleration from velocity sequence"""
    # velocity_sequence shape: (seq_length-1, 33, 2)
    acceleration = np.diff(velocity_sequence, axis=0)  # (seq_length-2, 33, 2)
    return acceleration

def calculate_side_acceleration_stats(acceleration_sequence, side_keypoints):
    """Calculate acceleration statistics for a body side"""
    # acceleration_sequence shape: (seq_length-2, 33, 2)
    side_acceleration = acceleration_sequence[:, side_keypoints, :]  # (seq_length-2, n_keypoints, 2)
    
    # Calculate magnitude of acceleration vectors
    acceleration_magnitudes = np.sqrt(np.sum(side_acceleration**2, axis=2))  # (seq_length-2, n_keypoints)
    
    # Aggregate statistics
    mean_acceleration = np.mean(acceleration_magnitudes)
    std_acceleration = np.std(acceleration_magnitudes)
    max_acceleration = np.max(acceleration_magnitudes)
    
    return mean_acceleration, std_acceleration, max_acceleration

def calculate_asymmetry_features(left_stats, right_stats):
    """Calculate asymmetry ratios between left and right sides"""
    left_mean_vel, left_std_vel, left_max_vel = left_stats['velocity']
    right_mean_vel, right_std_vel, right_max_vel = right_stats['velocity']
    
    left_mean_acc, left_std_acc, left_max_acc = left_stats['acceleration']
    right_mean_acc, right_std_acc, right_max_acc = right_stats['acceleration']
    
    # Velocity asymmetry ratios
    vel_mean_ratio = left_mean_vel / (right_mean_vel + 1e-8)  # Avoid division by zero
    vel_std_ratio = left_std_vel / (right_std_vel + 1e-8)
    vel_max_ratio = left_max_vel / (right_max_vel + 1e-8)
    
    # Acceleration asymmetry ratios
    acc_mean_ratio = left_mean_acc / (right_mean_acc + 1e-8)
    acc_std_ratio = left_std_acc / (right_std_acc + 1e-8)
    acc_max_ratio = left_max_acc / (right_max_acc + 1e-8)
    
    return [vel_mean_ratio, vel_std_ratio, vel_max_ratio, 
            acc_mean_ratio, acc_std_ratio, acc_max_ratio]

def extract_dynamic_features(df):
    dynamic_features_list = []
    
    for idx, row in df.iterrows():
        skeleton_seq = np.array(row['Skeleton_Sequence'])


        skeleton_seq = skeleton_seq.reshape(skeleton_seq.shape[0], 33, 2)

        # Calculate velocity and acceleration
        velocity = calculate_velocity(skeleton_seq)
        acceleration = calculate_acceleration(velocity)
        
        # Calculate statistics for each side
        left_vel_stats = calculate_side_velocity_stats(velocity, LEFT_SIDE_KEYPOINTS)
        right_vel_stats = calculate_side_velocity_stats(velocity, RIGHT_SIDE_KEYPOINTS)
        
        left_acc_stats = calculate_side_acceleration_stats(acceleration, LEFT_SIDE_KEYPOINTS)
        right_acc_stats = calculate_side_acceleration_stats(acceleration, RIGHT_SIDE_KEYPOINTS)
        
        # Create feature dictionaries
        left_stats = {'velocity': left_vel_stats, 'acceleration': left_acc_stats}
        right_stats = {'velocity': right_vel_stats, 'acceleration': right_acc_stats}
        
        # Calculate asymmetry features
        asymmetry_features = calculate_asymmetry_features(left_stats, right_stats)
        
        # Combine all dynamic features
        dynamic_features = [
            *left_vel_stats, *right_vel_stats,
            *left_acc_stats, *right_acc_stats,
            *asymmetry_features
        ]
        
        dynamic_features_list.append(dynamic_features)
    
    return np.array(dynamic_features_list)

# Helper function to extract dynamic features for a single sequence
def extract_dynamic_features_single(skeleton_seq):
    """Extract dynamic features for a single skeleton sequence"""
    # Calculate velocity and acceleration
    velocity = calculate_velocity(skeleton_seq)
    acceleration = calculate_acceleration(velocity)
    
    # Calculate statistics for each side
    left_vel_stats = calculate_side_velocity_stats(velocity, LEFT_SIDE_KEYPOINTS)
    right_vel_stats = calculate_side_velocity_stats(velocity, RIGHT_SIDE_KEYPOINTS)
    
    left_acc_stats = calculate_side_acceleration_stats(acceleration, LEFT_SIDE_KEYPOINTS)
    right_acc_stats = calculate_side_acceleration_stats(acceleration, RIGHT_SIDE_KEYPOINTS)
    
    # Create feature dictionaries
    left_stats = {
        'velocity': left_vel_stats,
        'acceleration': left_acc_stats
    }
    right_stats = {
        'velocity': right_vel_stats,
        'acceleration': right_acc_stats
    }
    
    # Calculate asymmetry features
    asymmetry_features = calculate_asymmetry_features(left_stats, right_stats)
    
    # Combine all dynamic features
    dynamic_features = [
        *left_vel_stats, *right_vel_stats,    # 6 velocity features
        *left_acc_stats, *right_acc_stats,    # 6 acceleration features  
        *asymmetry_features                   # 6 asymmetry ratios
    ]
    
    return np.array(dynamic_features)

# Function to create feature mask for top N features per exercise (BOTH static and dynamic)
def create_top_n_mask(n_features_per_exercise, feature_importance_df):

    # Create feature name to index mapping
    keypoint_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    feature_to_index = {}
    for i, name in enumerate(keypoint_names):
        feature_to_index[f'{name}_mean_x'] = (i, 'mean_x')
        feature_to_index[f'{name}_mean_y'] = (i, 'mean_y')
        feature_to_index[f'{name}_var_x'] = (i, 'var_x')
        feature_to_index[f'{name}_var_y'] = (i, 'var_y')

    # Dynamic feature names and their indices (they come after the 132 static features)
    dynamic_feature_names = [
        'left_side_mean_velocity', 'left_side_std_velocity', 'left_side_max_velocity',
        'right_side_mean_velocity', 'right_side_std_velocity', 'right_side_max_velocity',
        'left_side_mean_acceleration', 'left_side_std_acceleration', 'left_side_max_acceleration',
        'right_side_mean_acceleration', 'right_side_std_acceleration', 'right_side_max_acceleration',
        'velocity_mean_asymmetry_ratio', 'velocity_std_asymmetry_ratio', 'velocity_max_asymmetry_ratio',
        'acceleration_mean_asymmetry_ratio', 'acceleration_std_asymmetry_ratio', 'acceleration_max_asymmetry_ratio'
    ]

    # Add dynamic features to the mapping (they start at index 132)
    for i, name in enumerate(dynamic_feature_names):
        feature_to_index[name] = (132 + i, 'dynamic')

    mask_dict = {}
    
    for exercise in ['E1', 'E2', 'E3', 'E4', 'E5']:
        # Get top N features for this exercise (both static and dynamic)
        top_features = feature_importance_df[
            feature_importance_df['exercise'] == exercise
        ].nlargest(n_features_per_exercise, 'importance')
        
        mask_dict[exercise] = {}
        
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            if feature_name in feature_to_index:
                feature_idx, feature_type = feature_to_index[feature_name]
                
                if feature_type == 'dynamic':
                    # For dynamic features, we store them with a special key
                    if 'dynamic' not in mask_dict[exercise]:
                        mask_dict[exercise]['dynamic'] = set()
                    mask_dict[exercise]['dynamic'].add(feature_idx - 132)  # Convert to dynamic feature index (0-17)
                else:
                    # For static features, use the original keypoint-based system
                    kp_idx = feature_idx
                    component = feature_type
                    if kp_idx not in mask_dict[exercise]:
                        mask_dict[exercise][kp_idx] = []
                    if component not in mask_dict[exercise][kp_idx]:
                        mask_dict[exercise][kp_idx].append(component)
    
    return mask_dict

# Combined feature extraction function with top-N masking for BOTH static and dynamic features
def extract_combined_features_with_masking(df, top_n_mask):
    features_list = []
    
    for idx, row in df.iterrows():
        skeleton_seq = row['Skeleton_Sequence']
        exercise_id = row['Exercise_Id']
        
        # Get feature mask configuration for this exercise
        feature_mask_config = top_n_mask.get(exercise_id, {})
        
        # ===== EXTRACT STATIC FEATURES WITH MASKING =====
        # Ensure skeleton_seq is 3D: (seq_length, 33, 2)
        if skeleton_seq.ndim == 2 and skeleton_seq.shape[1] == 66:
            skeleton_seq = skeleton_seq.reshape(skeleton_seq.shape[0], 33, 2)
        
        # Calculate means and variances for all keypoints first
        flattened = skeleton_seq.reshape(len(skeleton_seq), -1)  # (seq_length, 66)
        all_means = np.mean(flattened, axis=0)  # 66 features
        all_variances = np.var(flattened, axis=0)  # 66 features
        
        # Apply granular masking - only keep specified static features
        final_means = np.zeros(66)
        final_variances = np.zeros(66)
        
        # For each keypoint in the mask configuration, keep specified components
        for kp_idx, components_to_keep in feature_mask_config.items():
            if kp_idx == 'dynamic':
                continue  # Skip dynamic features for now
            
            # Each keypoint has 2 positions in the mean/variance arrays
            mean_x_idx = kp_idx * 2
            mean_y_idx = kp_idx * 2 + 1
            var_x_idx = kp_idx * 2
            var_y_idx = kp_idx * 2 + 1
            
            if 'mean_x' in components_to_keep:
                final_means[mean_x_idx] = all_means[mean_x_idx]
            if 'mean_y' in components_to_keep:
                final_means[mean_y_idx] = all_means[mean_y_idx]
            if 'var_x' in components_to_keep:
                final_variances[var_x_idx] = all_variances[var_x_idx]
            if 'var_y' in components_to_keep:
                final_variances[var_y_idx] = all_variances[var_y_idx]
        
        static_features = np.concatenate([final_means, final_variances])
        
        # ===== EXTRACT DYNAMIC FEATURES WITH MASKING =====
        dynamic_features_all = extract_dynamic_features_single(skeleton_seq)

        # Dynamic feature names and their indices (they come after the 132 static features)
        dynamic_feature_names = [
            'left_side_mean_velocity', 'left_side_std_velocity', 'left_side_max_velocity',
            'right_side_mean_velocity', 'right_side_std_velocity', 'right_side_max_velocity',
            'left_side_mean_acceleration', 'left_side_std_acceleration', 'left_side_max_acceleration',
            'right_side_mean_acceleration', 'right_side_std_acceleration', 'right_side_max_acceleration',
            'velocity_mean_asymmetry_ratio', 'velocity_std_asymmetry_ratio', 'velocity_max_asymmetry_ratio',
            'acceleration_mean_asymmetry_ratio', 'acceleration_std_asymmetry_ratio', 'acceleration_max_asymmetry_ratio'
        ]
        
        # Apply masking to dynamic features
        dynamic_features_masked = np.zeros(len(dynamic_feature_names))
        if 'dynamic' in feature_mask_config:
            dynamic_indices_to_keep = feature_mask_config['dynamic']
            for idx in dynamic_indices_to_keep:
                if idx < len(dynamic_features_all):
                    dynamic_features_masked[idx] = dynamic_features_all[idx]
        
        # ===== COMBINE STATIC AND DYNAMIC FEATURES =====
        combined_features = np.concatenate([static_features, dynamic_features_masked])
        features_list.append(combined_features)
    
    return np.array(features_list)
