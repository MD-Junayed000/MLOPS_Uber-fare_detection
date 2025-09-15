from datetime import datetime
import pendulum

from airflow.sdk import DAG                      # Airflow 3 public authoring API
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

from data_processing import preprocess_data      # import functions only
from model_training import train_model
from model_evaluation import evaluate_model

tz = pendulum.timezone("Asia/Dhaka")

with DAG(
    dag_id="mlops_pipeline",
    start_date=pendulum.datetime(2025, 9, 6, tz=tz),
    schedule="@daily",                           # Airflow 3 uses `schedule`
    catchup=False,
    default_args={"owner": "airflow", "retries": 1},
    tags=["mlops"],
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    process_data = PythonOperator(
        task_id="process_data",
        python_callable=preprocess_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    start >> process_data >> train >> evaluate >> end
