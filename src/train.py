# src/train.py
import os, json, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow, mlflow.sklearn

os.makedirs("models", exist_ok=True)

# Resolve tracking URI: inside Docker we have MLFLOW_TRACKING_URI=http://mlflow:5000
# On your host machine, fall back to http://localhost:5000
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("iris-demo")

df = pd.read_csv("data/processed/iris_processed.csv")
X = df.drop(columns=["target"])
y = df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="rf-baseline"):
    clf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    mlflow.log_param("n_estimators", clf.n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

# Save DVC-tracked artifacts
joblib.dump(clf, "models/model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

print("Saved models/model.pkl  accuracy:", acc)
