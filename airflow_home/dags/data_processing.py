from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"

def preprocess_data():
    raw_csv = _data_dir() / "raw_data.csv"
    out_csv = _data_dir() / "processed_data.csv"

    # Define the correct column names
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

    # Load the dataset, set header=None to prevent pandas from using the first row as columns
    df = pd.read_csv(raw_csv, header=None, names=columns)

    # Print column names to debug
    print(f"Original columns: {df.columns}")

    # Apply scaling only to the selected columns
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

    # If the 'species' column exists, retain it
    if "species" in df.columns:
        df_scaled["species"] = df["species"]

    # Ensure the output directory exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write the processed data to a new CSV
    df_scaled.to_csv(out_csv, index=False)
    print(f"[preprocess_data] wrote {out_csv}")
