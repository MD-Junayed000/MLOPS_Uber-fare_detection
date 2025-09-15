from datetime import datetime
from pathlib import Path
import pendulum
import joblib

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

from model_training import train_model

def _data_dir():
    return Path(__file__).resolve().parent / "data"

def monitor_model():
    """Return the next task id based on a simple metric or model presence."""
    model_path = _data_dir() / "model.pkl"
    if not model_path.exists():
        return "retrain_model"
    # dummy threshold check (replace with your metric retrieval)
    metric = 0.80
    return "model_ok" if metric >= 0.85 else "retrain_model"

with DAG(
    dag_id="monitor_model_dag",
    start_date=pendulum.datetime(2025, 9, 6, tz="Asia/Dhaka"),
    schedule="@hourly",                 # Airflow 3 keyword
    catchup=False,
    tags=["monitoring"],
) as dag:
    decide = BranchPythonOperator(task_id="decide", python_callable=monitor_model)

    model_ok = EmptyOperator(task_id="model_ok")
    retrain_model = PythonOperator(task_id="retrain_model", python_callable=train_model)

    decide >> [model_ok, retrain_model]
