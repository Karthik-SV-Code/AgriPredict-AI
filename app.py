"""Flask app for crop-yield prediction and simple what-if analysis."""

from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

saved_data = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
model = saved_data["model"] if saved_data else None
encoder = saved_data["encoder"] if saved_data else None
feature_columns = saved_data["feature_columns"] if saved_data else None
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None

CATEGORICAL_FEATURES = ["Area", "Item"]
NUMERICAL_FEATURES = [
    "Year", "average_rain_fall_mm_per_year",
    "pesticides_tonnes", "avg_temp"
]


def make_input(form):
    """Build one input row using the same columns used during training."""
    return pd.DataFrame([{
        "Area": form["area"],
        "Item": form["crop"],
        "Year": int(form["year"]),
        "average_rain_fall_mm_per_year": float(form["rainfall"]),
        "pesticides_tonnes": float(form["pesticides"]),
        "avg_temp": float(form["temperature"])
    }])


def prepare_input(sample):
    """Apply the saved encoder and arrange columns in training order."""
    categories = encoder.transform(sample[CATEGORICAL_FEATURES])
    category_columns = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    category_data = pd.DataFrame(
        categories.toarray() if hasattr(categories, "toarray") else categories,
        columns=category_columns
    )
    numerical_data = sample[NUMERICAL_FEATURES].reset_index(drop=True)
    combined = pd.concat([category_data, numerical_data], axis=1)
    return combined.reindex(columns=feature_columns, fill_value=0)


def get_prediction(sample):
    return float(model.predict(prepare_input(sample))[0])


def render_page(**kwargs):
    return render_template("index.html", metrics=metrics, **kwargs)


@app.route("/")
def home():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_page(error="Train the model first by running: python train_model.py")

    try:
        sample = make_input(request.form)
        predicted_yield = get_prediction(sample)
        values = sample.iloc[0].to_dict()

        return render_page(
            prediction=predicted_yield,
            tonnes=predicted_yield / 10000,
            submitted=values,
            scenario=values,
            baseline=predicted_yield
        )
    except (ValueError, KeyError) as error:
        return render_page(error=f"Please check the entered values: {error}")


@app.route("/simulate", methods=["POST"])
def simulate():
    if model is None:
        return render_page(error="Train the model first by running: python train_model.py")

    try:
        # Original inputs are kept fixed except for the three scenario variables.
        original = {
            "area": request.form["area"],
            "crop": request.form["crop"],
            "year": request.form["year"],
            "rainfall": request.form["base_rainfall"],
            "pesticides": request.form["base_pesticides"],
            "temperature": request.form["base_temperature"]
        }
        scenario_form = {
            **original,
            "rainfall": request.form["scenario_rainfall"],
            "pesticides": request.form["scenario_pesticides"],
            "temperature": request.form["scenario_temperature"]
        }

        baseline_input = make_input(original)
        scenario_input = make_input(scenario_form)
        baseline = get_prediction(baseline_input)
        scenario_yield = get_prediction(scenario_input)

        return render_page(
            prediction=baseline,
            tonnes=baseline / 10000,
            submitted=baseline_input.iloc[0].to_dict(),
            scenario=scenario_input.iloc[0].to_dict(),
            baseline=baseline,
            scenario_yield=scenario_yield,
            difference=scenario_yield - baseline
        )
    except (ValueError, KeyError) as error:
        return render_page(error=f"Please check the scenario values: {error}")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
