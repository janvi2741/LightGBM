import pandas as pd
import joblib
import os
import glob
from sklearn.metrics import classification_report, roc_auc_score

# Load trained LightGBM model
print("Loading model...")
model = joblib.load("lightgbm_ids_model.pkl")
print("Model loaded successfully.\n")

# Ask user for CSV path
csv_path = input("Enter path to CSV file or directory for prediction: ").strip()

# Handle directory vs file input
if os.path.isdir(csv_path):
    # If directory provided, list available CSV files
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {csv_path}")
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    # Ask user to select a file
    while True:
        try:
            choice = int(input("\nSelect file number (or 0 to use cleaned dataset): "))
            if choice == 0:
                csv_path = "CICIDS2017_cleaned.csv"
                break
            elif 1 <= choice <= len(csv_files):
                csv_path = csv_files[choice-1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Load dataset
print(f"\nLoading dataset from: {os.path.basename(csv_path)}...")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Data loaded, shape: {df.shape}")

# Detect label column
possible_labels = ['Label', 'label', ' Label', 'Label ', ' Label ', 'CLASS', 'AttackType']
label_col = None
for col in possible_labels:
    if col in df.columns:
        label_col = col
        break

# If not found, search for any column containing 'label'
if label_col is None:
    for col in df.columns:
        if 'label' in str(col).lower():
            label_col = col
            break

if label_col is None:
    # If no common label found, assume last column is label
    label_col = df.columns[-1]
    print(f"No standard label column found. Using last column: {label_col}")
else:
    print(f"Label column detected: {label_col}")

# Convert to binary: BENIGN=0, All attacks=1
df['LabelBinary'] = df[label_col].apply(lambda x: 0 if str(x).upper()=='BENIGN' else 1)

# Check number of rows per class
counts = df['LabelBinary'].value_counts()
print(f"Rows per class: {counts.to_dict()}")

# Create balanced subset
if len(counts) == 1:
    print("Warning: Only one class present. Using all available samples.")
    df_subset = df
else:
    n_samples = min(counts.min(), 2500) if counts.min() > 0 else counts.max()
    
    samples = []
    for class_val in [0, 1]:
        if counts.get(class_val, 0) > 0:
            class_df = df[df['LabelBinary'] == class_val]
            sample_size = min(len(class_df), n_samples)
            samples.append(class_df.sample(n=sample_size, random_state=42))
    
    df_subset = pd.concat(samples, ignore_index=True) if samples else df

print(f"Balanced subset created, shape: {df_subset.shape}")

# Prepare features and labels
X = df_subset.drop(columns=[label_col, 'LabelBinary'], errors='ignore')
y = df_subset['LabelBinary']

print(f"Features shape: {X.shape} Labels shape: {y.shape}")

# Ensure feature compatibility with trained model
print("Preparing features for prediction...")

# Convert all features to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill NaN values
X = X.fillna(0)

# Replace infinite values
import numpy as np
X.replace([np.inf, -np.inf], 0, inplace=True)

print(f"Feature matrix shape: {X.shape}")

# Run predictions
print("Running predictions...")
try:
    y_pred = model.predict(X)
    print("Predictions completed.\n")
except Exception as e:
    print(f"Error during prediction: {e}")
    print("This might be due to feature mismatch between training and test data.")
    raise

# Save predictions
df_subset['Predicted'] = y_pred
df_subset.to_csv("predictions_output.csv", index=False)
print("Predictions saved to 'predictions_output.csv'.")

# Evaluate only if both classes exist
if len(y.unique()) > 1:
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    try:
        roc = roc_auc_score(y, y_pred)
        print(f"ROC AUC: {roc:.4f}")
    except Exception as e:
        print(f"ROC AUC could not be calculated: {e}")
else:
    print("\nOnly one class present. Skipping classification report and ROC AUC.")

