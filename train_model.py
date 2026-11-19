"""Train and evaluate the AgriPredict AI Random Forest model."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "yield_df.csv"
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

CATEGORICAL_FEATURES = ["Area", "Item"]
NUMERICAL_FEATURES = [
    "Year", "average_rain_fall_mm_per_year",
    "pesticides_tonnes", "avg_temp"
]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "hg/ha_yield"


def prepare_features(data, encoder, feature_columns):
    """Convert category names to numbers and keep columns in training order."""
    categories = encoder.transform(data[CATEGORICAL_FEATURES])
    category_columns = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    category_data = pd.DataFrame(
        categories.toarray() if hasattr(categories, "toarray") else categories,
        columns=category_columns,
        index=data.index
    )
    numerical_data = data[NUMERICAL_FEATURES].reset_index(drop=True)
    combined = pd.concat(
        [category_data.reset_index(drop=True), numerical_data], axis=1
    )
    return combined.reindex(columns=feature_columns, fill_value=0)


def main():
    # Load data and separate input features (X) from the target (y).
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET]

    # Keep 20% of the records aside for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Learn how to convert crop and region names into numeric columns.
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(X_train[CATEGORICAL_FEATURES])

    X_train_ready = prepare_features(X_train, encoder, (
        list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        + NUMERICAL_FEATURES
    ))
    feature_columns = list(X_train_ready.columns)
    X_test_ready = prepare_features(X_test, encoder, feature_columns)

    # Train a Random Forest using the prepared training data.
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train_ready, y_train)
    predictions = model.predict(X_test_ready)

    # Calculate regression evaluation metrics.
    metrics = {
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 2),
        "test_records": int(len(y_test))
    }

    # Save the trained model and the category encoder for later predictions.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "encoder": encoder,
        "feature_columns": feature_columns
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\nAGRI PREDICT AI - MODEL PERFORMANCE")
    print(f"R² Score : {metrics['r2']}")
    print(f"MAE      : {metrics['mae']:,.2f} hg/ha")
    print(f"RMSE     : {metrics['rmse']:,.2f} hg/ha")
    print(f"Test rows: {metrics['test_records']}")
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
