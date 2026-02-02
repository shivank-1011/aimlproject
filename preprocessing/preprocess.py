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
    
    # Coerce all potential numeric columns
    numeric_candidates = [c for c in df.columns if c != 'Section']
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace negative subject scores with NaN
    subject_cols = ['Maths', 'SESD', 'AIML', 'FSD', 'DVA']
    for col in subject_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Apply IQR Capping (Winsorization)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    # 5. Impute missing values with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    df = df.fillna(0)

    # 6. One-hot encode 'Section' column
    df = pd.get_dummies(df, columns=['Section'], prefix='Section', drop_first=False)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # 7. Apply Standard Scaling
    scaler = StandardScaler()
    scale_cols = df.select_dtypes(include=[np.number]).columns
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # 8. Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 9. Save output
    print(f"Saving train data to {output_train_path}...")
    train_df.to_csv(output_train_path, index=False)
    print(f"Saving test data to {output_test_path}...")
    test_df.to_csv(output_test_path, index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    RAW_DATA = 'data/raw/nststudentsrawdata.csv'
    TRAIN_PATH = 'data/processed/train_processed.csv'
    TEST_PATH = 'data/processed/test_processed.csv'
    
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    
    preprocess_data(RAW_DATA, TRAIN_PATH, TEST_PATH)