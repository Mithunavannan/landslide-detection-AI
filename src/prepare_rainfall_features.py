
from pathlib import Path
import pandas as pd

# Project folder is the parent of src/
PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "rainfall_daily_district_summary.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "rainfall_daily_features.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Rainfall file not found:\n{INPUT_FILE}\n\n"
            "Make sure the file is inside data/processed/."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "date",
        "rainfall_mean_mm",
        "rainfall_max_mm",
        "rainfall_min_mm",
        "station_count",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}\n"
            f"Columns found: {list(df.columns)}"
        )

    # Convert dates and measurements to the correct data types.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "rainfall_mean_mm",
        "rainfall_max_mm",
        "rainfall_min_mm",
        "station_count",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid dates and invalid rainfall values.
    df = df.dropna(subset=["date", "rainfall_mean_mm"]).copy()
    df = df[df["rainfall_mean_mm"] >= 0].copy()

    # Use calendar dates, without a time component.
    df["date"] = df["date"].dt.normalize()

    # Ensure there is only one district summary per date.
    # If duplicate dates exist, aggregate them.
    daily = (
        df.groupby("date", as_index=True)
        .agg(
            rainfall_mean_mm=("rainfall_mean_mm", "mean"),
            rainfall_max_mm=("rainfall_max_mm", "max"),
            rainfall_min_mm=("rainfall_min_mm", "min"),
            station_count=("station_count", "max"),
        )
        .sort_index()
    )

    if daily.empty:
        raise ValueError("No valid rainfall rows remain after cleaning.")

    # Add missing calendar dates as NaN.
    # We do NOT assume a missing record means zero rainfall.
    full_dates = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D",
        name="date",
    )
    daily = daily.reindex(full_dates)
    daily.index.name = "date"

    # Short-term rainfall features.
    daily["rainfall_previous_day_mm"] = (
        daily["rainfall_mean_mm"].shift(1)
    )

    # Rolling windows require every day in the window to have a value.
    # If any day is missing, that window's feature stays NaN.
    daily["rain_3day_sum_mm"] = (
        daily["rainfall_mean_mm"]
        .rolling(window=3, min_periods=3)
        .sum()
    )

    daily["rain_7day_sum_mm"] = (
        daily["rainfall_mean_mm"]
        .rolling(window=7, min_periods=7)
        .sum()
    )

    daily["rain_14day_sum_mm"] = (
        daily["rainfall_mean_mm"]
        .rolling(window=14, min_periods=14)
        .sum()
    )

    daily["rain_3day_max_mm"] = (
        daily["rainfall_mean_mm"]
        .rolling(window=3, min_periods=3)
        .max()
    )

    daily["rain_7day_max_mm"] = (
        daily["rainfall_mean_mm"]
        .rolling(window=7, min_periods=7)
        .max()
    )

    # Calendar features may help the model account for seasonality.
    daily["month"] = daily.index.month
    daily["day_of_year"] = daily.index.dayofyear

    daily = daily.reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_FILE, index=False)

    print("\nRainfall feature preparation completed.")
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Number of calendar days: {len(daily)}")
    print(f"First date: {daily['date'].min().date()}")
    print(f"Last date:  {daily['date'].max().date()}")
    print("\nPreview:")
    print(daily.head(10).to_string(index=False))

    print(
        "\nNote: NaN values in rainfall or rolling features are intentional "
        "where observations are missing or the rolling window is incomplete."
    )


if __name__ == "__main__":
    main()
