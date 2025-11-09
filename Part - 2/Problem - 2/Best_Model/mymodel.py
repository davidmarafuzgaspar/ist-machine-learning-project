import numpy as np
import joblib
from feature_extraction import extract_combined_features_with_masking

# Load the trained model artifacts
artifacts = joblib.load("champion.pkl")

# Extract components from artifacts
model = artifacts['model']
feature_mask = artifacts['feature_mask']
exercise_encoder = artifacts['exercise_encoder']

def predict(Xtest):
    """
    Predict impaired side for test data
    
    Parameters:
    - Xtest: DataFrame with columns ['Patient_Id', 'Exercise_Id', 'Skeleton_Sequence']
    
    Returns:
    - y_pred: NumPy array of predictions (0 for left, 1 for right) for each PATIENT
    """
    # Extract features using the same masking
    X_features = extract_combined_features_with_masking(Xtest, feature_mask)
    
    # Encode exercises
    exercise_encoded = exercise_encoder.transform(Xtest[['Exercise_Id']])
    X_combined = np.concatenate([X_features, exercise_encoded], axis=1)
    
    # Get sequence-level predictions AND probabilities
    sequence_preds = model.predict(X_combined)
    sequence_probs = model.predict_proba(X_combined)
    
    # Aggregate to patient-level using enhanced majority voting
    patient_predictions = {}
    patient_probabilities = {}
    
    for idx, row in Xtest.iterrows():
        patient_id = row['Patient_Id']
        if patient_id not in patient_predictions:
            patient_predictions[patient_id] = []
            patient_probabilities[patient_id] = []
        patient_predictions[patient_id].append(sequence_preds[idx])
        patient_probabilities[patient_id].append(sequence_probs[idx])
    
    # Apply enhanced majority voting for final patient predictions
    final_predictions = {}
    patient_ids = sorted(patient_predictions.keys())
    
    for patient_id in patient_ids:
        votes = patient_predictions[patient_id]
        probs = patient_probabilities[patient_id]
        
        # Count votes for each class
        vote_counts = np.bincount(votes)
        
        # Check for tie
        if len(vote_counts) > 1 and vote_counts[0] == vote_counts[1]:
            # Tie detected - use average probability to break it
            avg_probs = np.mean(probs, axis=0)
            final_label = np.argmax(avg_probs)
            print(f"Tie detected for patient {patient_id}. Breaking with probabilities: {avg_probs}")
        else:
            # Normal majority vote
            final_label = np.argmax(vote_counts)
        
        final_predictions[patient_id] = final_label
    
    # Convert to array in the same order as unique patient IDs
    y_pred_patient = np.array([final_predictions[pid] for pid in patient_ids])
    
    return y_pred_patient