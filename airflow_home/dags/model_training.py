from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def _data_dir():
    return Path(__file__).resolve().parent / "data"

def train_model():
    processed_csv = _data_dir() / "processed_data.csv"
    model_path = _data_dir() / "model.pkl"

    df = pd.read_csv(processed_csv)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]  # make sure this column exists

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

    acc = accuracy_score(y_te, model.predict(X_te))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[train_model] saved model to {model_path} | accuracy: {acc:.3f}")
