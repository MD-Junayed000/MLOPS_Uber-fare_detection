#!/usr/bin/env bash
set -euo pipefail

# Expect AIRFLOW__DATABASE__SQL_ALCHEMY_CONN etc. passed in via container env

echo "[init] migrating Airflow DB..."
airflow db migrate

# Create admin user idempotently
airflow users create \
  --username "${AIRFLOW_ADMIN_USER:-admin}" \
  --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com || true

echo "[start] launching Airflow scheduler & webserver..."
airflow scheduler &  # background
airflow webserver &  # background

echo "[start] launching ZenML API on :8237 ..."
exec uvicorn zenml.zen_server.zen_server_api:app --host 0.0.0.0 --port 8237
