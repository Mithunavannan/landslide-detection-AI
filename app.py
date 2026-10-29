
from pathlib import Path
import json
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


# ============================================================
# 1. PROJECT PATHS
# app.py must be in the project root beside data/, models/,
# templates/, and static/.
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "wayanad_rainfall_prototype.joblib"
)
METRICS_PATH = PROJECT_DIR / "models" / "prototype_metrics.json"
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "rainfall_daily_district_summary.csv"
)

app = Flask(__name__)

# ============================================================
# 2. LOAD THE MODEL ONCE
# ============================================================
model_bundle = None
model = None
feature_columns = []
high_rainfall_threshold = None
load_error = None

print("\n" + "=" * 60)
print("TERRAWATCH MODEL INITIALIZATION")
print("=" * 60)
print(f"Application file : {Path(__file__).resolve()}")
print(f"Project directory: {PROJECT_DIR}")
print(f"Model file       : {MODEL_PATH}")
print(f"Model exists     : {MODEL_PATH.is_file()}")

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file does not exist at: {MODEL_PATH}"
        )

    # Load the trained model bundle.
    model_bundle = joblib.load(MODEL_PATH)

    if not isinstance(model_bundle, dict):
        raise TypeError(
            "Unexpected model file format. "
            "Expected a dictionary containing model and feature_columns."
        )

    if "model" not in model_bundle:
        raise KeyError(
            "The model bundle has no 'model' entry. "
            "Run python src/train_xgboost.py to recreate it."
        )

    model = model_bundle["model"]
    feature_columns = model_bundle.get("feature_columns", [])
    high_rainfall_threshold = model_bundle.get(
        "high_rainfall_threshold_mm"
    )

    if not feature_columns:
        raise ValueError(
            "The saved model bundle contains no feature_columns."
        )

    print("Model loaded     : SUCCESS")
    print(f"Model type       : {type(model).__name__}")
    print(f"Features         : {feature_columns}")
    print(f"Rainfall threshold: {high_rainfall_threshold}")

except Exception as exc:
    model_bundle = None
    model = None
    feature_columns = []
    load_error = f"{type(exc).__name__}: {exc}"

    print("Model loaded     : FAILED")
    print(f"Loading error    : {load_error}")
    traceback.print_exc()

print("=" * 60 + "\n")


# ============================================================
# 3. LOAD DASHBOARD METRICS
# ============================================================
def load_metrics():
    if not METRICS_PATH.is_file():
        return {}

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        app.logger.warning("Could not read metrics file: %s", exc)
        return {}


# ============================================================
# 4. LOAD DATASET SUMMARY
# ============================================================
def load_data_summary():
    summary = {
        "rows": None,
        "start_date": None,
        "end_date": None,
        "rainfall_mean": None,
        "max_rainfall": None,
    }

    if not DATA_PATH.is_file():
        app.logger.warning("Rainfall dataset not found: %s", DATA_PATH)
        return summary

    try:
        df = pd.read_csv(DATA_PATH)
        summary["rows"] = int(len(df))

        if "date" in df.columns:
            dates = pd.to_datetime(
                df["date"], errors="coerce"
            ).dropna()

            if not dates.empty:
                summary["start_date"] = dates.min().strftime("%d %b %Y")
                summary["end_date"] = dates.max().strftime("%d %b %Y")

        if "rainfall_mean_mm" in df.columns:
            values = pd.to_numeric(
                df["rainfall_mean_mm"], errors="coerce"
            ).dropna()

            if not values.empty:
                summary["rainfall_mean"] = round(
                    float(values.mean()), 2
                )

        if "rainfall_max_mm" in df.columns:
            values = pd.to_numeric(
                df["rainfall_max_mm"], errors="coerce"
            ).dropna()

            if not values.empty:
                summary["max_rainfall"] = round(
                    float(values.max()), 2
                )

    except Exception as exc:
        app.logger.exception("Could not read rainfall dataset: %s", exc)

    return summary


# ============================================================
# 5. DASHBOARD PAGE
# ============================================================
@app.route("/")
def index():
    metrics = load_metrics()
    summary = load_data_summary()

    return render_template(
        "index.html",
        model_ready=(model is not None),
        load_error=load_error,
        metrics=metrics,
        summary=summary,
        threshold=high_rainfall_threshold,
    )


# ============================================================
# 6. MODEL INFERENCE API
# ============================================================
@app.route("/api/predict", methods=["POST"])
def predict():
    # Check the model loaded successfully at startup.
    if model is None:
        return jsonify({
            "error": (
                "The XGBoost model could not be loaded. "
                f"Details: {load_error or 'Unknown loading error'}"
            )
        }), 503

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Invalid request. Expected JSON input."
        }), 400

    # Validate required values.
    try:
        rainfall_max = float(payload["rainfall_max_mm"])
        rainfall_min = float(payload["rainfall_min_mm"])
        station_count = float(payload["station_count"])

    except KeyError as exc:
        return jsonify({
            "error": f"Missing required input: {exc.args[0]}"
        }), 400

    except (TypeError, ValueError):
        return jsonify({
            "error": "All rainfall inputs must be valid numbers."
        }), 400

    # Validate numeric ranges.
    if not all(
        pd.notna(value)
        for value in [rainfall_max, rainfall_min, station_count]
    ):
        return jsonify({
            "error": "Inputs must be finite numeric values."
        }), 400

    if rainfall_max < 0 or rainfall_min < 0:
        return jsonify({
            "error": "Rainfall values cannot be negative."
        }), 400

    if rainfall_min > rainfall_max:
        return jsonify({
            "error": "Minimum rainfall cannot exceed maximum rainfall."
        }), 400

    if (
        station_count <= 0
        or not station_count.is_integer()
        or station_count > 10000
    ):
        return jsonify({
            "error": "Station count must be a positive whole number."
        }), 400

    # The saved model expects these exact feature names.
    input_values = {
        "rainfall_max_mm": rainfall_max,
        "rainfall_min_mm": rainfall_min,
        "station_count": int(station_count),
    }

    try:
        # Build features in the exact order stored with the model.
        missing = [
            col for col in feature_columns
            if col not in input_values
        ]

        if missing:
            raise ValueError(
                "The model expects features not supplied by the form: "
                + ", ".join(missing)
            )

        sample = pd.DataFrame(
            [[input_values[col] for col in feature_columns]],
            columns=feature_columns,
        )

        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        classes = list(model.classes_)

        # Find the probability corresponding to class 1.
        if 1 not in classes:
            raise ValueError(
                "The loaded model does not contain high-rainfall class 1."
            )

        high_rainfall_probability = float(
            probabilities[classes.index(1)]
        )

        if prediction == 1:
            label = "High-rainfall proxy"
        else:
            label = "Below high-rainfall proxy"

        return jsonify({
            "success": True,
            "prediction": prediction,
            "label": label,
            "probability": high_rainfall_probability,
            "threshold_mm": high_rainfall_threshold,
            "inputs": input_values,
            "notice": (
                "This is a high-rainfall demonstration, not a "
                "landslide prediction or emergency warning."
            ),
        }), 200

    except Exception as exc:
        app.logger.exception("Model inference failed")
        return jsonify({
            "error": f"Model inference failed: {type(exc).__name__}: {exc}"
        }), 500


# ============================================================
# 7. HEALTH CHECK — OPEN /api/health TO CHECK MODEL STATUS
# ============================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "app": "TerraWatch",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_file_exists": MODEL_PATH.is_file(),
        "features": feature_columns,
        "error": load_error,
    }), 200


# ============================================================
# 8. START LOCAL DEVELOPMENT SERVER
# ============================================================
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
