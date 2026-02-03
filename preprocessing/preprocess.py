import pandas as pd
import numpy as np
import json
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def parse_json_column(df, column_name, prefix):
    """Parses a column containing JSON strings into separate columns."""
    def safe_json_loads(x):
        try:
            if isinstance(x, str):
                # Fix unquoted keys in JSON string
                x = re.sub(r'([a-zA-Z0-9_]+):', r'"\1":', x)
                return json.loads(x)
            return {}
        except Exception:
            return {}

    expanded_data = df[column_name].apply(safe_json_loads).apply(pd.Series)
    expanded_data = expanded_data.add_prefix(f'{prefix}_')
    return pd.concat([df, expanded_data], axis=1).drop(columns=[column_name])

def preprocess_data(input_path, output_train_path, output_test_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    display_df = df.copy()

    # 1. Drop unnecessary columns
    cols_to_drop = ['id', 'student_name']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. Extract Section from URN
    def extract_section(urn):
        if not isinstance(urn, str):
            return 'Unknown'
        match = re.search(r'2024[-]?([A-Z])', urn)
        if match:
            return match.group(1)
        return 'Unknown'

    df['Section'] = df['URN'].apply(extract_section)
    df = df.drop(columns=['URN'])

    # 3. Parse JSON-like columns
    if 'topic_wise_accuracy' in df.columns:
        df = parse_json_column(df, 'topic_wise_accuracy', 'Acc')
    if 'time_spent_per_topic' in df.columns:
        df = parse_json_column(df, 'time_spent_per_topic', 'Time')

    # 4. Handle invalid values and outliers
    # Coerce numeric columns
    numeric_candidates = [c for c in df.columns if c != 'Section']
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace negative scores with NaN
    subject_cols = ['Maths', 'SESD', 'AIML', 'FSD', 'DVA']
    for col in subject_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    
    # Generate Synthetic Target 'PlacementStatus' (1 if avg > 75, else 0)
    temp_subjects = df[subject_cols].fillna(0)
    df['PlacementStatus'] = (temp_subjects.mean(axis=1) > 75).astype(int)
    print(f"Target distribution:\n{df['PlacementStatus'].value_counts()}")

    # Apply IQR Capping (Winsorization)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'PlacementStatus']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    # 5. Impute missing values with mean
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    df = df.fillna(0)

    # 6. One-hot encode 'Section'
    df = pd.get_dummies(df, columns=['Section'], prefix='Section', drop_first=False)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # 7. Split data (before scaling to fix leakage)
    X = df.drop(columns=['PlacementStatus'])
    y = df['PlacementStatus']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 8. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 9. Save output
    # Concatenate X and y for saving (or save X and y separately, but typically CSVs have both)
    print(f"Saving train data to {output_train_path}...")
    train_out = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    train_out.to_csv(output_train_path, index=False)
    
    print(f"Saving test data to {output_test_path}...")
    test_out = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    test_out.to_csv(output_test_path, index=False)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    RAW_DATA = 'data/raw/nststudentsrawdata.csv'
    TRAIN_PATH = 'data/processed/train_processed.csv'
    TEST_PATH = 'data/processed/test_processed.csv'
    
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    
    preprocess_data(RAW_DATA, TRAIN_PATH, TEST_PATH)