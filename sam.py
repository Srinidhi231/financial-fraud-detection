# ===============================
# 1. Import libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse
import json
import os
from joblib import dump

# ===============================
# 2. Load data
# ===============================
# (CSV files are in the same directory)
# NOTE: To avoid running out of memory on large datasets, we load a subset
#       of rows for local training. This keeps training fast and stable while
#       still producing a model consistent with the Streamlit app.
NROWS_TRAIN = 200_000  # adjust if your machine can handle more/less

train_transaction = pd.read_csv('train_transaction.csv', nrows=NROWS_TRAIN)
train_identity    = pd.read_csv('train_identity.csv',    nrows=NROWS_TRAIN)
test_transaction  = pd.read_csv('test_transaction.csv')
test_identity     = pd.read_csv('test_identity.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Merge transaction & identity data
train_df = train_transaction.merge(train_identity, on='TransactionID', how='left')
test_df  = test_transaction.merge(test_identity,  on='TransactionID', how='left')

# Normalize column names (e.g., id-01 -> id_01) to match between train and test
train_df.columns = [c.replace('-', '_') for c in train_df.columns]
test_df.columns  = [c.replace('-', '_') for c in test_df.columns]

# ===============================
# 3. Prepare features and target
# ===============================
y = train_df['isFraud']
X = train_df.drop(columns=['isFraud'])

# For Kaggle test set
X_test_final = test_df.copy()

# ===============================
# 4. Encode categorical variables consistently
# ===============================
# Fit mappings on train and apply to test for stable encodings
object_cols = X.select_dtypes(include=['object']).columns.tolist()
categorical_mappings = {}
for col in object_cols:
    labels, uniques = pd.factorize(X[col], sort=True)
    mapping = {cat: idx for idx, cat in enumerate(uniques)}
    categorical_mappings[col] = mapping
    X[col] = labels
    if col in X_test_final.columns:
        X_test_final[col] = X_test_final[col].map(mapping).fillna(-1).astype(int)
    else:
        # If column missing in test, add as default -1
        X_test_final[col] = -1

# Ensure test has the same columns and order as train
missing_in_test = [c for c in X.columns if c not in X_test_final.columns]
for c in missing_in_test:
    X_test_final[c] = -999
extra_in_test = [c for c in X_test_final.columns if c not in X.columns]
if extra_in_test:
    X_test_final = X_test_final.drop(columns=extra_in_test)
X_test_final = X_test_final[X.columns]

# Fill missing values (RandomForest doesn’t like NaN)
X = X.fillna(-999)
X_test_final = X_test_final.fillna(-999)

# ===============================
# 5. Train/Validation split
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6. Fit RandomForest
# ===============================
model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=10,       # limit depth to avoid overfitting
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
model.fit(X_train, y_train)

# ===============================
# 7. Evaluate on validation set
# ===============================
val_preds = model.predict_proba(X_val)[:,1]   # probability of fraud
print("Validation ROC AUC:", roc_auc_score(y_val, val_preds))
# Also report validation accuracy at threshold 0.5
val_labels = (val_preds >= 0.5).astype(int)
print("Validation Accuracy:", accuracy_score(y_val, val_labels))

# ===============================
# 8. Predict on Kaggle test set
# ===============================
test_preds = model.predict_proba(X_test_final)[:,1]

# ===============================
# 9. Create submission file
# ===============================
sample_submission['isFraud'] = test_preds
sample_submission.to_csv('submission.csv', index=False)
print("submission.csv file created!")

# ===============================
# 10. Optional: Predict on a user-provided CSV
# ===============================
def preprocess_for_prediction(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [c.replace('-', '_') for c in df.columns]
    # Apply categorical mappings
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[col] = -1
    # Add any missing numeric columns and drop extras
    missing_in_df = [c for c in X.columns if c not in df.columns]
    for c in missing_in_df:
        df[c] = -999
    extra_cols = [c for c in df.columns if c not in X.columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)
    df = df[X.columns]
    df = df.fillna(-999)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_csv', type=str, default=None, help='Path to a CSV to classify (combined features like merged test).')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV for predictions when using --predict_csv')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save model and preprocessing artifacts')
    args, unknown = parser.parse_known_args()

    # Save artifacts (model + preprocessing metadata)
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, 'fraud_model.joblib')
    mappings_path = os.path.join(args.save_dir, 'categorical_mappings.json')
    features_path = os.path.join(args.save_dir, 'feature_columns.json')
    dump(model, model_path)
    with open(mappings_path, 'w', encoding='utf-8') as f:
        # ensure keys are strings and values are ints
        serializable_mappings = {col: {str(k): int(v) for k, v in mapping.items()} for col, mapping in categorical_mappings.items()}
        json.dump(serializable_mappings, f)
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(list(X.columns), f)
    print(f"Artifacts saved to {args.save_dir}: fraud_model.joblib, categorical_mappings.json, feature_columns.json")

    if args.predict_csv is not None:
        user_df = pd.read_csv(args.predict_csv)
        user_X = preprocess_for_prediction(user_df)
        user_probs = model.predict_proba(user_X)[:,1]
        user_labels = (user_probs >= 0.5).astype(int)
        out = pd.DataFrame({'fraud_probability': user_probs, 'isFraud_pred': user_labels})
        # If the input contains TransactionID, include it
        if 'TransactionID' in user_df.columns:
            out = pd.concat([user_df[['TransactionID']].reset_index(drop=True), out], axis=1)
        out.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
