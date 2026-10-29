
from pathlib import Path
import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_DIR / "models" / "wayanad_rainfall_prototype.joblib"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Model not found. Run python src/train_xgboost.py first."
    )

saved = joblib.load(MODEL_FILE)
model = saved["model"]
feature_columns = saved["feature_columns"]
threshold = saved["high_rainfall_threshold_mm"]

print("\nWAYANAD RAINFALL PROTOTYPE TEST")
print("=" * 40)
print(f"High-rainfall threshold: {threshold:.2f} mm")
print("Enter station rainfall summary values.")
print("These are the same types of inputs used by the model.\n")

try:
    rainfall_max = float(input("Maximum station rainfall (mm): "))
    rainfall_min = float(input("Minimum station rainfall (mm): "))
    station_count = float(input("Number of reporting stations: "))

    if rainfall_max < 0 or rainfall_min < 0 or station_count <= 0:
        raise ValueError("Rainfall cannot be negative; station count must be positive.")

    if rainfall_min > rainfall_max:
        raise ValueError("Minimum rainfall cannot exceed maximum rainfall.")

    sample = pd.DataFrame([{
        "rainfall_max_mm": rainfall_max,
        "rainfall_min_mm": rainfall_min,
        "station_count": station_count,
    }], columns=feature_columns)

    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])

    print("\nRESULT")
    print("-" * 40)
    if prediction == 1:
        print("Prediction: HIGH-RAINFALL PROXY")
    else:
        print("Prediction: BELOW HIGH-RAINFALL PROXY")

    print(f"Model probability for high-rainfall class: {probability:.1%}")
    print(f"Training-derived threshold: {threshold:.2f} mm")

    print(
        "\nIMPORTANT: This is a rainfall-category demonstration, "
        "not a landslide-risk or landslide-warning prediction."
    )

except ValueError as error:
    print(f"\nInvalid input: {error}")
