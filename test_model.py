import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

# ----------------------------
# Load trained LightGBM model
# ----------------------------
print("Loading model...")
model = joblib.load("lightgbm_ids_model.pkl")
print("Model loaded successfully.\n")

# ----------------------------
# Load dataset
# ----------------------------
print("Loading dataset...")
df = pd.read_csv("CICIDS2017_cleaned.csv", low_memory=False)
df.columns = df.columns.str.strip()
print("Data loaded, shape:", df.shape)

# ----------------------------
# Detect label column
# ----------------------------
label_col = None
for col in df.columns:
    if col.lower() == 'label':
        label_col = col
        break

if label_col is None:
    raise ValueError("No 'Label' column found!")

# ----------------------------
# Map labels: BENIGN = 0, all attacks = 1
# ----------------------------
df['LabelBinary'] = df[label_col].apply(lambda x: 0 if str(x).upper()=='BENIGN' else 1)
df_both = df[df['LabelBinary'].isin([0,1])]

# ----------------------------
# Count rows per class
# ----------------------------
counts = df_both['LabelBinary'].value_counts()
print("Rows per class:", counts.to_dict())

# ----------------------------
# Handle single-class case
# ----------------------------
if len(counts) < 2:
    print("\nWarning: Only one class is present in the dataset. Metrics like ROC AUC or classification report may be invalid.\n")
    df_subset = df_both
else:
    n_samples = min(counts.min(), 2500)
    print(f"Sampling up to {n_samples} rows per class for balanced subset.")
    samples = []
    for class_label in [0,1]:
        class_df = df_both[df_both['LabelBinary']==class_label]
        if len(class_df) == 0:
            print(f"Warning: No data for class {class_label}, skipping this class.")
            continue
        n = min(len(class_df), n_samples)
        samples.append(class_df.sample(n=n, random_state=42))
    df_subset = pd.concat(samples)
    print("Balanced subset created, shape:", df_subset.shape)

# ----------------------------
# Prepare features and label
# ----------------------------
X = df_subset.drop(columns=[label_col,'LabelBinary'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # convert all to numeric and fill NaNs
y = df_subset['LabelBinary']

print("Features shape:", X.shape, "Labels shape:", y.shape)

# ----------------------------
# Make predictions
# ----------------------------
print("Running predictions...")
y_pred = model.predict(X)

try:
    y_prob = model.predict_proba(X)[:,1]
except AttributeError:
    y_prob = None

print("Predictions completed.\n")

# ----------------------------
# Evaluate model
# ----------------------------
if len(counts) < 2:
    print("Only one class present. Skipping classification report and ROC AUC.\n")
else:
    print("Classification Report:\n", classification_report(y, y_pred))
    if y_prob is not None:
        print("ROC AUC:", roc_auc_score(y, y_prob))

# ----------------------------
# Optional: Show first 10 predictions
# ----------------------------
print("\nFirst 10 predictions:\n", y_pred[:10])


