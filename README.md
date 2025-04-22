
# Customer Churn: End-to-End Machine Learning Pipeline


pipeline_build->run_pipeline->experiment_track(mlflow)->test->deployer(local/cloud)->monitor(flask/streamlit)->retrain_check->schedualer(local/prefect)->run_pipeline(cicd)

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000



# ml_customer_churn

[![check.yml](https://github.com/Hsinghsudwal/ml_customer_churn/actions/workflows/check.yml/badge.svg)](https://github.com/Hsinghsudwal/ml_customer_churn/actions/workflows/check.yml)
[![publish.yml](https://github.com/Hsinghsudwal/ml_customer_churn/actions/workflows/publish.yml/badge.svg)](https://github.com/Hsinghsudwal/ml_customer_churn/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://Hsinghsudwal.github.io/ml_customer_churn/)
[![License](https://img.shields.io/github/license/Hsinghsudwal/ml_customer_churn)](https://github.com/Hsinghsudwal/ml_customer_churn/blob/main/LICENCE.txt)
[![Release](https://img.shields.io/github/v/release/Hsinghsudwal/ml_customer_churn)](https://github.com/Hsinghsudwal/ml_customer_churn/releases)

TODO.

# Installation

Use the package manager [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

# Usage

```bash
uv run ml_customer_churn
```
