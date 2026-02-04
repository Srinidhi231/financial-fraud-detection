import json
import pandas as pd
from joblib import load
import shap

from app import preprocess, load_artifacts


def main():
    print("Loading artifacts...")
    model, categorical_mappings, feature_columns = load_artifacts(
        "fraud_model.joblib", "categorical_mappings.json", "feature_columns.json"
    )
    print("Model type:", type(model))
    print("Model n_features_in_:", getattr(model, "n_features", None), getattr(model, "n_features_in_", None))

    print("Loading test data...")
    df = pd.read_csv("my_test_input.csv")
    print("Raw test shape:", df.shape)

    X = preprocess(df, categorical_mappings, feature_columns)
    print("Transformed X shape:", X.shape)

    print("Creating TreeExplainer on retrained model with X as background...")
    explainer = shap.TreeExplainer(model, data=X)

    print("Computing SHAP values on a sample of up to 50 rows...")
    sample = X.sample(min(50, len(X)), random_state=0)
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        print("shap_vals is list of length", len(shap_vals))
        for i, arr in enumerate(shap_vals):
            print(f"class {i} shap_vals shape:", getattr(arr, "shape", None))
    else:
        print("shap_vals shape:", getattr(shap_vals, "shape", None))


if __name__ == "__main__":
    main()


