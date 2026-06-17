# run_pipeline.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib

import config
from io_utils import list_signal_files, load_numeric_file
from preprocess import detrend_signal, normalize_columns
from feature_engineering import features_from_array
from modeling import MODELS
from modeling_severity import train_severity_regressor


def infer_label(path):
    name = os.path.basename(path).lower()
    if "pt" in name:
        return config.LABEL_PD
    if "co" in name:
        return config.LABEL_CONTROL
    return None


def load_severity_table():
    path = os.path.join(config.BASE_DIR, "data", "severity.csv")
    if not os.path.exists(path):
        print("⚠ WARNING: 'severity.csv' NOT FOUND — skipping severity prediction.")
        return None
    return pd.read_csv(path)


def main():

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("Loading gait files...")
    files = list_signal_files(config.DATA_DIR)
    print("Found", len(files), "files.")

    rows = []

    # Extract features
    for f in files:
        arr = load_numeric_file(f)
        if arr is None:
            continue

        arr = detrend_signal(arr)
        arr = normalize_columns(arr)

        feats = features_from_array(arr)
        label = infer_label(f)

        rows.append({"file": f, "label": label, **feats})

    df = pd.DataFrame(rows).dropna(subset=["label"])

    # Classification data
    X = df.drop(columns=["file", "label"])
    y = df["label"].astype(int)

    # Imputer
    # Missing value imputer: mean / median / mode (most_frequent)
    imputer = SimpleImputer(strategy=config.IMPUTE_STRATEGY)

# Apply to feature matrix
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


    # Split
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X_imp, y, df["file"], test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )

    # Train classification models
    metrics_list = []
    preds_dict = {}
    model_dict = {}

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        model_dict[model_name] = model
        preds_dict[model_name] = preds

        metrics_list.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds)
        })

    # Save classification metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(config.METRICS_OUTPUT, index=False)

    # Pick best model
    best_row = metrics_df.sort_values(["accuracy", "f1_score"], ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = model_dict[best_model_name]
    

    # Save predictions
    pred_df = pd.DataFrame({
    "file": [os.path.basename(f).split("_")[0] for f in f_test.values],
    "true_label": y_test.values,
    "predicted_label": preds_dict[best_model_name]
    })
    pred_df.to_csv(config.PREDICTIONS_OUTPUT, index=False)


    # Save best model
    joblib.dump({
        "model_name": best_model_name,
        "model": best_model,
        "imputer": imputer,
        "feature_columns": X.columns.tolist()
    }, config.MODEL_OUTPUT)

    print("Best model:", best_model_name)
    # -------------------------
    # CONFUSION MATRIX + HEATMAP (ADDED)
    # -------------------------
    y_pred_best = preds_dict[best_model_name]

    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
       cm,
       annot=True,
       fmt='d',
       cmap='coolwarm',   # colorful heatmap
       linewidths=1,
       linecolor='black',
       xticklabels=["Control", "Parkinson"],
       yticklabels=["Control", "Parkinson"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {best_model_name}")

    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print("Confusion matrix saved at:", cm_path)

    # -------------------------
    # CLASSIFICATION REPORT (ADDED)
    # -------------------------
    report = classification_report(y_test, y_pred_best)

    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    # -------------------------
    # SEVERITY PREDICTION
    # -------------------------
    severity_table = load_severity_table()
    if severity_table is None:
        return

    # Merge severity data
    df["file_name"] = df["file"].apply(lambda x: os.path.basename(x))
    merged = df.merge(severity_table, left_on="file_name", right_on="file_name")
    if merged.empty:
        print("⚠ No matching severity entries found.")
        return

    X_sev = merged[X.columns]
    y_sev = merged["UPDRS"]

    Xs_train, Xs_test, ys_train, ys_test, fs_train, fs_test = train_test_split(
        X_sev, y_sev, merged["file"], test_size=0.20, random_state=42
    )

    sev_model, sev_preds, sev_metrics = train_severity_regressor(
        Xs_train, ys_train, Xs_test, ys_test
    )

    # Save severity metrics
    pd.DataFrame([sev_metrics]).to_csv(
        os.path.join(config.RESULTS_DIR, "severity_metrics.csv"), index=False
    )

    # Save severity predictions
    pd.DataFrame({
    "file": [os.path.basename(f).split("_")[0] for f in fs_test.values],
    "true_severity": ys_test.values,
    "predicted_severity": sev_preds
    }).to_csv(
    os.path.join(config.RESULTS_DIR, "severity_predictions.csv"), index=False
    )


    print("Severity prediction complete.")


if __name__ == "__main__":
    main()
    
