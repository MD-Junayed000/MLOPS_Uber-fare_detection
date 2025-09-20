import pandas as pd, joblib, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/processed/iris_processed.csv")
X = df.drop(columns=["target"]); y = df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))
joblib.dump(clf, "models/model.pkl")
with open("metrics.json", "w") as f: json.dump({"accuracy": acc}, f)
print("Saved models/model.pkl  accuracy:", acc)
