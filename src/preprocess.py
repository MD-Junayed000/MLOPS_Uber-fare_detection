import pandas as pd, os
from sklearn.datasets import load_iris
os.makedirs("data/processed", exist_ok=True)
df = load_iris(as_frame=True).frame
df.to_csv("data/processed/iris_processed.csv", index=False)
print("Wrote data/processed/iris_processed.csv")
