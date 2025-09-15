from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report

def _data_dir():
    return Path(__file__).resolve().parent / "data"

def evaluate_model():
    model = joblib.load(_data_dir() / "model.pkl")
    df = pd.read_csv(_data_dir() / "processed_data.csv")

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    preds = model.predict(X)
    print("=== Model Evaluation Report ===")
    print(classification_report(y, preds))
