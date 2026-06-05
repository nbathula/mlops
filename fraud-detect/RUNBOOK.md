# Fraud Detect — MLOps Runbook

End-to-end reference for setting up, training, deploying, and operating the fraud detection system.

---

## Architecture Overview

```
Raw CSV (S3)
    │
    ▼
Ingestion (PySpark / pandas)
    │
    ▼
Feature Engineering
  - amount_log, amount_zscore
  - hour_of_day, is_night
    │
    ▼
XGBoost Training (SMOTE for class imbalance)
    │
    ▼
MLflow Experiment Tracking (Databricks)
    │
    ▼
Model Registry (Databricks Unity Catalog)
  workspace.default.fraud-detection-model
    │
    ▼
Databricks Model Serving Endpoint
  fraud-detect-endpoint
```

---

## Prerequisites

| Tool | Install |
|---|---|
| Python 3.11+ | `brew install python` |
| AWS CLI | `brew install awscli` |
| Databricks CLI v2 | `brew install databricks/tap/databricks` |
| GitHub CLI | `brew install gh` |

---

## 1. First-Time Setup

### Clone and create virtual environment
```bash
git clone https://github.com/nbathula/mlops.git
cd mlops/fraud-detect
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Copy and fill in credentials
```bash
cp .env.example .env
# Edit .env and fill in:
# DATABRICKS_HOST, DATABRICKS_TOKEN
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

### Authenticate Databricks CLI
```bash
databricks configure --host $DATABRICKS_HOST --token $DATABRICKS_TOKEN
```

---

## 2. Upload Training Data to S3

The raw dataset (`data/creditcard.csv`) must be uploaded once:

```bash
source .env && export $(grep -v '^#' .env | xargs)
aws s3 cp data/creditcard.csv s3://fraud-detect-artifacts/raw/creditcard.csv
```

---

## 3. Set Up Databricks Secrets

AWS credentials are stored in a Databricks secret scope so notebooks can access S3 without hardcoding keys:

```bash
source .env && export $(grep -v '^#' .env | xargs)
bash scripts/setup_secrets.sh
```

This creates the scope `fraud-detect` with keys:
- `aws_access_key_id`
- `aws_secret_access_key`

---

## 4. Deploy the Databricks Asset Bundle

Syncs notebooks and the job definition to your Databricks workspace:

```bash
databricks bundle deploy
```

To verify:
```bash
databricks bundle validate
```

> Re-run this any time you change `notebooks/`, `databricks.yml`, or `pipelines/`.

---

## 5. Train the Model

### Option A — Locally (registers model to Databricks)
```bash
source .env && export $(grep -v '^#' .env | xargs)
MLFLOW_TRACKING_URI=databricks \
MLFLOW_EXPERIMENT_NAME="/Users/naga.bathula@gmail.com/fraud-detection" \
MODEL_NAME=workspace.default.fraud-detection-model \
python -m src.training.train
```

### Option B — Via Databricks Job (requires compute-enabled workspace)
```bash
databricks bundle run fraud_detect_training_pipeline
```

Training output:
- Experiment: `https://dbc-2bb4a99b-a265.cloud.databricks.com/ml/experiments/...`
- Registered model: `workspace.default.fraud-detection-model`
- Target metrics: ROC-AUC ≥ 0.95 to auto-promote to Production

---

## 6. Deploy the Serving Endpoint

Creates (or updates) the `fraud-detect-endpoint` on Databricks Model Serving:

```bash
source .env && export $(grep -v '^#' .env | xargs)
MODEL_NAME=workspace.default.fraud-detection-model \
MODEL_VERSION=1 \        # change to latest version
python scripts/deploy_serving.py
```

The script waits until the endpoint is `READY` and prints the invocation URL.

---

## 7. Test the Endpoint

```bash
curl -X POST \
  "$DATABRICKS_HOST/serving-endpoints/fraud-detect-endpoint/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
      "V6": 0.46,  "V7": 0.24,  "V8": 0.10, "V9": 0.36, "V10": 0.09,
      "V11": -0.55,"V12": -0.62,"V13": -0.99,"V14": -0.31,"V15": 1.47,
      "V16": -0.47,"V17": 0.21, "V18": 0.03,"V19": 0.40, "V20": 0.25,
      "V21": -0.02,"V22": 0.28, "V23": -0.11,"V24": 0.07,"V25": 0.13,
      "V26": -0.19,"V27": 0.13, "V28": -0.02,
      "amount_log": 0.0, "amount_zscore": -0.34,
      "hour_of_day": 10, "is_night": 0
    }]
  }'
# Expected: {"predictions": [0]}  (0=legit, 1=fraud)
```

> Input fields are **engineered features** — not raw `Amount`/`Time`.

---

## 8. CI/CD Pipeline

Three GitHub Actions workflows trigger automatically on push to `main`:

| Workflow | Triggers on | Does what |
|---|---|---|
| `test.yml` | Every push / PR | Runs pytest |
| `train.yml` | `src/**` or `data/**` changes | Trains model, registers to Databricks UC |
| `deploy.yml` | After Train succeeds | Updates serving endpoint to new model version |
| `bundle.yml` | `notebooks/**`, `databricks.yml`, `pipelines/**` changes | Runs `databricks bundle deploy` |

### Required GitHub Secrets

Go to **Settings → Secrets → Actions** and ensure these exist:

| Secret | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | Personal access token |
| `MLFLOW_EXPERIMENT_NAME` | `/Users/your@email.com/fraud-detection` |
| `AWS_ACCESS_KEY_ID` | For S3 access |
| `AWS_SECRET_ACCESS_KEY` | For S3 access |
| `S3_BUCKET` | `fraud-detect-artifacts` |

---

## 9. Cost Management

| Resource | When it costs | How to stop |
|---|---|---|
| Serving endpoint | While provisioned | Delete via API or Databricks UI |
| Clusters | While running | Auto-terminate after 60 min idle |
| SQL Warehouse | While running | Auto-stops when idle |

### Delete the serving endpoint (to stop billing)
```bash
source .env && export $(grep -v '^#' .env | xargs)
curl -X DELETE "$DATABRICKS_HOST/api/2.0/serving-endpoints/fraud-detect-endpoint" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN"
```

### Recreate it when needed
```bash
MODEL_NAME=workspace.default.fraud-detection-model MODEL_VERSION=1 \
python scripts/deploy_serving.py
```

---

## 10. Project Structure

```
fraud-detect/
├── .github/workflows/
│   ├── test.yml          # Run pytest on every push
│   ├── train.yml         # Train + register model
│   ├── deploy.yml        # Update serving endpoint
│   └── bundle.yml        # Sync notebooks to Databricks
├── api/                  # Flask REST API (local serving alternative)
├── data/                 # Raw CSV (gitignored)
├── databricks.yml        # Asset Bundle root config
├── frontend/             # Streamlit dashboard
├── notebooks/            # Databricks notebook pipeline
│   ├── 01_ingest.py
│   ├── 02_features.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   └── 05_drift.py
├── pipelines/
│   └── databricks_workflow.yml   # Databricks job DAG
├── scripts/
│   ├── setup_secrets.sh          # Create Databricks secret scope
│   └── deploy_serving.py         # Create/update serving endpoint
├── src/
│   ├── ingestion/ingest.py
│   ├── features/feature_engineering.py
│   ├── training/train.py
│   ├── evaluation/evaluate.py
│   └── monitoring/drift.py
├── terraform/            # AWS S3 + IAM infrastructure
└── tests/
```

---

## Key Databricks Links

- Workspace: `https://dbc-2bb4a99b-a265.cloud.databricks.com`
- MLflow experiments: Workspace → Experiments → `fraud-detection`
- Model registry: Workspace → Models → `workspace.default.fraud-detection-model`
- Serving endpoints: Workspace → Serving
