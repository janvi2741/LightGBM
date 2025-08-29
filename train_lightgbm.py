
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
import joblib

# Load cleaned dataset
df = pd.read_csv("CICIDS2017_cleaned.csv", low_memory=False)

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Detect label column
label_col = None
for col in df.columns:
    if col.lower() == 'label':
        label_col = col
        break

if label_col is None:
    raise ValueError("No 'Label' column found in the dataset!")

# Separate features and label
X = df.drop(columns=[label_col])
y = df[label_col]

# Binary classification: 0 = BENIGN, 1 = ATTACK
y = y.apply(lambda s: 0 if str(s).upper()=='BENIGN' else 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# LightGBM model
model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    class_weight='balanced',
    n_jobs=-1
)

# Train the model (older LightGBM compatible)
print("Training LightGBM model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),  # early stopping
        lgb.log_evaluation(period=100)            # log every 100 rounds
    ]
)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Save the model
joblib.dump(model, "lightgbm_ids_model.pkl")
print("Model saved as lightgbm_ids_model.pkl")
