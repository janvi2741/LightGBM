#!/usr/bin/env python3
"""
Simple test runner for the IDS model
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def run_quick_test():
    """Run a quick test of the model"""
    print("Loading model and data...")
    
    # Load model
    model = joblib.load("lightgbm_ids_model.pkl")
    
    # Load test data
    df = pd.read_csv("CICIDS2017_cleaned.csv", nrows=5000)
    df.columns = df.columns.str.strip()
    
    # Find label column
    label_col = None
    for col in df.columns:
        if 'label' in str(col).lower():
            label_col = col
            break
    
    if not label_col:
        print("No label column found!")
        return
    
    # Prepare data
    X = df.drop(columns=[label_col])
    y_true = df[label_col].apply(lambda x: 0 if str(x).upper()=='BENIGN' else 1)
    
    # Clean features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Show results
    print(f"\nTest Results on {len(X)} samples:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    if len(np.unique(y_true)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
    else:
        print("Only one class in test data")

if __name__ == "__main__":
    run_quick_test()
