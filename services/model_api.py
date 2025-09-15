from pathlib import Path
from typing import Any, List

import joblib
from flask import Flask, jsonify, request

# Try to import pandas; if it isn’t installed the API will still run,
# but predictions will emit the warning.
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None  # type: ignore

app = Flask(__name__)
_model: Any | None = None

def _model_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "airflow_home"
        / "dags"
        / "data"
        / "model.pkl"
    )

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(_model_path())
    return _model

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": (
            "Model API is running. Send a POST request to /predict with "
            "a JSON body containing a 'features' array."
        )
    })

@app.post("/predict")
def predict() -> tuple:
    payload = request.get_json(silent=True)
    if not payload or "features" not in payload:
        return jsonify({"error": "Missing 'features' in input data"}), 400

    features: List[float] = payload["features"]
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": (
                "'features' must be a list of four numeric values: "
                "[sepal_length, sepal_width, petal_length, petal_width]"
            )
        }), 400

    try:
        model = get_model()
        # If pandas is available and the model was trained with feature names,
        # wrap the input in a DataFrame with matching column names
        if pd is not None and hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            if len(features) == len(cols):
                input_df = pd.DataFrame([features], columns=cols)
                prediction = model.predict(input_df)[0]
            else:
                prediction = model.predict([features])[0]
        else:
            prediction = model.predict([features])[0]
    except Exception as exc:
        return jsonify({"error": f"Model prediction failed: {exc}"}), 500

    # Ensure the prediction is JSON serialisable
    try:
        prediction = prediction.item()
    except Exception:
        pass

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
