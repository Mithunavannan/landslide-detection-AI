
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
INPUT_FILE = DATA_DIR / "rainfall_daily_district_summary.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "wayanad_rainfall_prototype.joblib"
METRICS_FILE = MODEL_DIR / "prototype_metrics.json"
LABELS_FILE = DATA_DIR / "rainfall_proxy_labels.csv"
IMPORTANCE_FILE = MODEL_DIR / "feature_importance.csv"

# We deliberately exclude rainfall_mean_mm from the input features.
# It is used only to define the high-rainfall proxy target.
FEATURE_COLUMNS = [
    "rainfall_max_mm",
    "rainfall_min_mm",
    "station_count",
]


def main():
    # -----------------------------------------------------
    # 2. Load the existing cleaned rainfall file
    # -----------------------------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}\n\n"
            "Check that rainfall_daily_district_summary.csv "
            "is inside data/processed/."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "date",
        "rainfall_mean_mm",
        "rainfall_max_mm",
        "rainfall_min_mm",
        "station_count",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.normalize()

    numeric_columns = [
        "rainfall_mean_mm",
        "rainfall_max_mm",
        "rainfall_min_mm",
        "station_count",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["date"] + numeric_columns
    ).copy()

    df = df[
        (df["rainfall_mean_mm"] >= 0)
        & (df["rainfall_max_mm"] >= 0)
        & (df["rainfall_min_mm"] >= 0)
        & (df["station_count"] > 0)
    ].copy()

    df = df.sort_values("date").drop_duplicates(
        subset=["date"], keep="last"
    ).reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} valid rows found. "
            "More rainfall observations are needed for this prototype."
        )

    # -----------------------------------------------------
    # 3. Chronological split BEFORE calculating the threshold
    # -----------------------------------------------------
    split_index = int(len(df) * 0.80)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Unable to split the dataset.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Define "unusually high rainfall" using only the training
    # period. The 90th percentile is a prototype rule, not a
    # scientifically validated landslide threshold.
    threshold = float(
        train_df["rainfall_mean_mm"].quantile(0.90)
    )

    train_df["high_rainfall_proxy"] = (
        train_df["rainfall_mean_mm"] >= threshold
    ).astype(int)

    test_df["high_rainfall_proxy"] = (
        test_df["rainfall_mean_mm"] >= threshold
    ).astype(int)

    y_train = train_df["high_rainfall_proxy"]
    y_test = test_df["high_rainfall_proxy"]

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    print("\nDataset summary")
    print("----------------")
    print(f"Valid observations: {len(df)}")
    print(f"Training rows:      {len(train_df)}")
    print(f"Testing rows:       {len(test_df)}")
    print(f"Training period:    {train_df['date'].min().date()} "
          f"to {train_df['date'].max().date()}")
    print(f"Testing period:     {test_df['date'].min().date()} "
          f"to {test_df['date'].max().date()}")
    print(f"High-rainfall threshold: {threshold:.2f} mm")
    print(f"Training class counts: {y_train.value_counts().to_dict()}")
    print(f"Testing class counts:  {y_test.value_counts().to_dict()}")

    if y_train.nunique() < 2:
        raise ValueError(
            "The training period produced only one class. "
            "The high-rainfall proxy cannot be trained."
        )

    # -----------------------------------------------------
    # 4. Train XGBoost
    # -----------------------------------------------------
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=(
            negative_count / positive_count
            if positive_count > 0
            else 1.0
        ),
    )

    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # 5. Evaluate on later observations
    # -----------------------------------------------------
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_purpose": "High-rainfall proxy demonstration only",
        "not_a_landslide_prediction_model": True,
        "label_definition": (
            "1 means rainfall_mean_mm is at or above the "
            "90th percentile of the training-period rainfall mean."
        ),
        "threshold_mm": threshold,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "test_proxy_positive_count": int(y_test.sum()),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "roc_auc": (
            float(roc_auc_score(y_test, probabilities))
            if y_test.nunique() == 2
            else None
        ),
        "features": FEATURE_COLUMNS,
        "warning": (
            "These metrics evaluate a rainfall proxy, not landslide "
            "occurrence. The proxy threshold is a demonstration rule, "
            "not a validated landslide trigger threshold."
        ),
    }

    # -----------------------------------------------------
    # 6. Save the generated proxy labels for transparency
    # -----------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    labelled = df.copy()
    labelled["high_rainfall_proxy"] = (
        labelled["rainfall_mean_mm"] >= threshold
    ).astype(int)
    labelled["proxy_threshold_mm"] = threshold
    labelled["label_type"] = "high_rainfall_proxy_not_landslide_label"

    labelled.to_csv(LABELS_FILE, index=False)

    # -----------------------------------------------------
    # 7. Save model, metrics, and feature importance
    # -----------------------------------------------------
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "high_rainfall_threshold_mm": threshold,
            "target_name": "high_rainfall_proxy",
            "not_a_landslide_predictor": True,
        },
        MODEL_FILE,
    )

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(IMPORTANCE_FILE, index=False)

    # -----------------------------------------------------
    # 8. Print results
    # -----------------------------------------------------
    print("\nPrototype classification report")
    print("--------------------------------")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Below proxy threshold",
                "High-rainfall proxy",
            ],
            zero_division=0,
        )
    )

    print("Evaluation metrics")
    print("------------------")
    print(json.dumps(metrics, indent=4))

    print("\nFeature importance")
    print("------------------")
    print(importance.to_string(index=False))

    print("\nSaved outputs")
    print("-------------")
    print(f"Proxy labels:     {LABELS_FILE}")
    print(f"Trained model:    {MODEL_FILE}")
    print(f"Metrics:          {METRICS_FILE}")
    print(f"Feature importances: {IMPORTANCE_FILE}")

    print(
        "\nIMPORTANT: This is a high-rainfall demonstration. "
        "Do not describe its metrics as landslide prediction accuracy."
    )


if __name__ == "__main__":
    main()
