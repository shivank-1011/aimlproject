import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import pickle
import os

def train_and_evaluate():
    TRAIN_PATH = 'data/processed/train_processed.csv'
    TEST_PATH = 'data/processed/test_processed.csv'
    MODEL_PATH = 'model/model.pkl'

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Separate Features and Target
    target_col = 'PlacementStatus'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    # Logistic Regression (Main Model)
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    
    print("\n--- Logistic Regression Performance ---")
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    print(f"Accuracy: {acc_lr:.4f}")
    print(f"Precision: {prec_lr:.4f}")
    print(f"Recall: {rec_lr:.4f}")
    print(f"Confusion Matrix:\n{cm_lr}")
    
    print(f"\nSaving Logistic Regression model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(lr_model, f)

    # Decision Tree (Comparison)
    print("\nTraining Decision Tree (Comparison)...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    y_pred_dt = dt_model.predict(X_test)
    
    print("\n--- Decision Tree Performance ---")
    acc_dt = accuracy_score(y_test, y_pred_dt)
    prec_dt = precision_score(y_test, y_pred_dt, zero_division=0)
    rec_dt = recall_score(y_test, y_pred_dt, zero_division=0)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    
    print(f"Accuracy: {acc_dt:.4f}")
    print(f"Precision: {prec_dt:.4f}")
    print(f"Recall: {rec_dt:.4f}")
    print(f"Confusion Matrix:\n{cm_dt}")

    # Summary Table
    print("\n=== Model Comparison Table ===")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 50)
    print(f"{'Logistic Regression':<20} {acc_lr:.4f}     {prec_lr:.4f}      {rec_lr:.4f}")
    print(f"{'Decision Tree':<20} {acc_dt:.4f}     {prec_dt:.4f}      {rec_dt:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    train_and_evaluate()