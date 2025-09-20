from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    "iris_pipeline",
    start_date=datetime(2025,1,1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command="cd /mlops && dvc pull || true"
    )

    dvc_repro = BashOperator(
        task_id="dvc_repro",
        bash_command="cd /mlops && dvc repro"
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command="cd /mlops && dvc push && git add . && git commit -m 'update model' || true"
    )

    dvc_pull >> dvc_repro >> dvc_push
