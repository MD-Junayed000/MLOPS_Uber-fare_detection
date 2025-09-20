from zenml import pipeline, step
import pandas as pd
from sklearn.datasets import load_iris

@step
def load_step() -> pd.DataFrame:
    return load_iris(as_frame=True).frame

@step
def count_step(df: pd.DataFrame) -> int:
    return len(df)

@pipeline
def iris_zen_pipeline():
    df = load_step()
    _ = count_step(df)

if __name__ == "__main__":
    iris_zen_pipeline()()
