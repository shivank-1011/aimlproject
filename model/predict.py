import pandas as pd
import pickle
import os
import numpy as np

def predict_students():
    MODEL_PATH = 'model/model.pkl'
    TEST_PATH = 'data/processed/test_processed.csv'
    OUTPUT_DIR = 'data/predictions'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'classification_report.csv')

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(TEST_PATH):
        print(f"Error: Test data not found at {TEST_PATH}")
        return

    print("Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print("Loading test data...")
    try:
        test_df = pd.read_csv(TEST_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Prepare features for prediction
    # We need to drop 'PlacementStatus' if it exists, as it's the target
    target_col = 'PlacementStatus'
    if target_col in test_df.columns:
        X_test = test_df.drop(columns=[target_col])
        actual_status = test_df[target_col]
    else:
        X_test = test_df
        actual_status = None

    print(f"Predicting for {len(X_test)} students...")
    
    # Get probabilities for class 1 (Placement)
    try:
        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            print("Model does not support predict_proba")
            return
    except Exception as e:
        print(f"Prediction error: {e}")
        return

    # Categorize
    # At-risk: < 40%
    # Average: 40% <= p < 80%
    # High-performing: >= 80%
    
    categories = []
    for p in probs:
        if p < 0.40:
            categories.append('At-risk')
        elif p < 0.80:
            categories.append('Average')
        else:
            categories.append('High-performing')

    # Create result DataFrame
    result_df = pd.DataFrame({
        'Predicted_Prob': probs,
        'Category': categories
    })
    
    if actual_status is not None:
        result_df['Actual_PlacementStatus'] = actual_status

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Saving predictions to {OUTPUT_PATH}...")
    result_df.to_csv(OUTPUT_PATH, index=False)
    
    # Summary
    print("\n--- Prediction Summary ---")
    print(result_df['Category'].value_counts())
    print("\nSample Output:")
    print(result_df.head())

if __name__ == "__main__":
    predict_students()
