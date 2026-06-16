# GCP ML Project — Customer Churn Prediction

End-to-end MLOps pipeline on Google Cloud Platform predicting customer churn using BigQuery, Vertex AI Model Registry, and Vertex AI Endpoints.

---

## Architecture

```
Local CSV Data
    │
    ▼
GCS Bucket (serious-bliss-256222-ml-data)
    │  raw/customers.csv
    │  raw/transactions.csv
    ▼
BigQuery (ml_churn dataset)
    │  customers table
    │  transactions table
    │  features table  ← SQL join + aggregations
    ▼
Training (local, Python)
    │  RandomForest + scikit-learn
    │  ROC-AUC logged to Vertex AI Experiments
    ▼
GCS Model Artifact  (models/churn/model.joblib)
    │
    ▼
Vertex AI Model Registry  (versioned, auditable)
    │
    ▼
Vertex AI Endpoint  (managed REST API for predictions)
```

---

## Project Structure

```
gcp-ml-project/
├── data/
│   ├── customers.csv          # Customer attributes + churn label
│   └── transactions.csv       # Transaction history per customer
├── scripts/
│   ├── upload_data.py         # Upload CSVs → GCS → BigQuery
│   ├── train_vertex.py        # Train locally, register to Vertex AI
│   ├── deploy_endpoint.py     # Deploy registered model to Vertex AI Endpoint
│   └── trainer/
│       └── task.py            # Training code (portable, runs anywhere)
├── terraform/
│   ├── main.tf                # GCS bucket, BigQuery, service account, APIs
│   ├── variables.tf
│   ├── outputs.tf
│   └── terraform.tfvars       # Project ID, region, bucket name
├── requirements.txt
└── README.md
```

---

## Features

| Feature | Source |
|---|---|
| `age` | customers table |
| `tenure_months` | customers table |
| `monthly_charge` | customers table |
| `num_products` | customers table |
| `support_calls` | customers table |
| `total_transactions` | aggregated from transactions |
| `total_spend` | aggregated from transactions |
| `avg_transaction` | aggregated from transactions |
| `label` (churned) | customers table |

---

## Prerequisites

| Tool | Install |
|---|---|
| Python 3.10+ | `brew install python` |
| Terraform | `brew install hashicorp/tap/terraform` |
| Google Cloud SDK | `brew install --cask google-cloud-sdk` |
| GitHub CLI | `brew install gh` |

---

## Setup

### 1. Authenticate GCP
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project serious-bliss-256222
```

### 2. Deploy Infrastructure
```bash
cd terraform
terraform init
terraform apply
```

Creates:
- GCS bucket with `raw/`, `processed/`, `models/` folders
- BigQuery dataset `ml_churn` with `customers` and `transactions` tables
- Service account `ml-workload-sa` with required roles
- Enables APIs: Storage, BigQuery, Vertex AI, Compute, Artifact Registry

### 3. Install Python Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Upload Data to GCS + BigQuery
```bash
python scripts/upload_data.py
```

Uploads `data/customers.csv` and `data/transactions.csv` to GCS, then loads them into BigQuery.

---

## Train + Deploy

### Train Model
```bash
python scripts/train_vertex.py
```

- Creates `ml_churn.features` table in BigQuery (SQL feature engineering)
- Trains RandomForest classifier locally
- Uploads `model.joblib` to GCS
- Registers model to **Vertex AI Model Registry** (versioned)
- Logs params + metrics to **Vertex AI Experiments**

### Deploy Endpoint
```bash
python scripts/deploy_endpoint.py
```

- Fetches latest model version from Vertex AI Model Registry
- Creates (or reuses) a Vertex AI Endpoint
- Deploys model — serves predictions via managed REST API

---

## Run Predictions

```python
from google.cloud import aiplatform

aiplatform.init(project="serious-bliss-256222", location="us-central1")

endpoint = aiplatform.Endpoint(
    "projects/761762714045/locations/us-central1/endpoints/3800079861117091840"
)

# Features: [age, tenure_months, monthly_charge, num_products,
#             support_calls, total_transactions, total_spend, avg_transaction]
prediction = endpoint.predict(instances=[[34, 24, 65.5, 2, 1, 5, 300.0, 60.0]])
print(prediction)  # 0 = no churn, 1 = churn
```

---

## Cost Management

| Resource | Cost | How to stop |
|---|---|---|
| Vertex AI Endpoint | ~$0.05/hr per replica | Delete endpoint when not in use |
| BigQuery | Per query (free tier: 1TB/mo) | No action needed |
| GCS | ~$0.02/GB/mo | Minimal for this dataset |
| Compute APIs | Only during training | No ongoing cost |

### Delete endpoint to stop billing
```bash
python -c "
from google.cloud import aiplatform
aiplatform.init(project='serious-bliss-256222', location='us-central1')
ep = aiplatform.Endpoint('projects/761762714045/locations/us-central1/endpoints/3800079861117091840')
ep.undeploy_all()
ep.delete()
print('Endpoint deleted')
"
```

---

## GCP Console Links

| Resource | Link |
|---|---|
| BigQuery | https://console.cloud.google.com/bigquery?project=serious-bliss-256222 |
| GCS Bucket | https://console.cloud.google.com/storage/browser/serious-bliss-256222-ml-data |
| Vertex AI Models | https://console.cloud.google.com/vertex-ai/models?project=serious-bliss-256222 |
| Vertex AI Endpoints | https://console.cloud.google.com/vertex-ai/endpoints?project=serious-bliss-256222 |
| Vertex AI Experiments | https://console.cloud.google.com/vertex-ai/experiments?project=serious-bliss-256222 |

---

## Future Improvements

### 1. Replace Manual Scripts with Proper GCP Services

| Current | Recommended GCP Service | Benefit |
|---|---|---|
| Manual CSV upload (`upload_data.py`) | Cloud Storage Transfer Service | Scheduled, monitored, retriable ingestion |
| Raw `.joblib` file in GCS | Artifact Registry | Versioned, scannable model + container artifacts |
| BigQuery SQL in Python script | Vertex AI Feature Store | Reusable, shareable features across models and teams |
| Manual `aiplatform.log_metrics()` | Vertex AI Experiments (full SDK) | Auto-log datasets, parameters, metrics, and lineage |
| Manual endpoint update | Vertex AI Model Monitoring | Automatic drift detection and alerting post-deployment |

---

### 2. CI/CD with Vertex AI Pipelines

Replace the current 3-script manual flow (`upload_data.py` → `train_vertex.py` → `deploy_endpoint.py`) with a **Vertex AI Pipeline** — a fully managed DAG that runs on GCP and is triggered automatically via GitHub Actions on push to `main`.

**Pipeline DAG:**

```
Step 1: ingest_data
        Upload CSV → GCS → BigQuery tables

Step 2: feature_engineering
        BigQuery SQL join → ml_churn.features table

Step 3: train_model
        Fetch features → train RandomForest → save artifact to GCS

Step 4: evaluate_model
        Compute ROC-AUC → pass if ≥ threshold, else fail pipeline

Step 5: register_model
        Upload artifact → Vertex AI Model Registry (new version)

Step 6: deploy_endpoint
        Deploy latest model version → Vertex AI Endpoint
```

**GitHub Actions trigger:**
```yaml
on:
  push:
    branches: [main]
    paths:
      - data/**
      - scripts/**
jobs:
  run-pipeline:
    steps:
      - run: python pipelines/churn_pipeline.py
```

This eliminates manual steps, adds a quality gate (evaluate_model), provides full audit trail, and enables automatic rollback if a new version underperforms.

---

### 3. Vertex AI AutoML Tabular

Instead of writing custom training code, **AutoML Tabular** automatically tries dozens of algorithms and hyperparameter combinations and picks the best model — no ML expertise required.

**What it replaces:**

| Current (`train_vertex.py`) | AutoML |
|---|---|
| Manually chose RandomForest | Tries RF, XGBoost, LightGBM, Neural Net, ensembles |
| Manually set `n_estimators=100` | Hyperparameter search is automatic |
| Wrote training code (~80 lines) | Zero training code |
| No feature importance | Built-in SHAP feature importance |

**AutoML training code (~20 lines):**

```python
from google.cloud import aiplatform

aiplatform.init(project="serious-bliss-256222", location="us-central1")

# Point AutoML at your BigQuery features table
dataset = aiplatform.TabularDataset.create(
    display_name="churn-dataset",
    bq_source="bq://serious-bliss-256222.ml_churn.features",
)

# Let AutoML find the best model — no algorithm choice needed
job = aiplatform.AutoMLTabularTrainingJob(
    display_name="churn-automl",
    optimization_prediction_type="classification",
    optimization_objective="maximize-au-roc",
)

model = job.run(
    dataset=dataset,
    target_column="label",
    budget_milli_node_hours=1000,   # 1 training hour budget
    model_display_name="churn-automl-model",
)
# model is auto-registered to Vertex AI Model Registry
```

**When to use AutoML vs Custom Training:**

| | AutoML | Custom (current) |
|---|---|---|
| Code to write | ~20 lines | ~80 lines |
| Algorithm choice | Automatic | Manual |
| Hyperparameter tuning | Built-in | Manual |
| Feature importance | Built-in (SHAP) | Manual |
| Cost | Higher (per node-hour) | Lower |
| Control | Low | Full |
| Best for | Baselines, non-ML teams | Production custom logic |

**Recommended approach:** Run AutoML first to get a benchmark AUC and understand feature importance, then decide if a custom model is worth the investment.

---

### 4. Additional Enhancements

- **Terraform remote state** — store `terraform.tfstate` in GCS instead of locally so the team can share infrastructure state
- **Model monitoring** — enable Vertex AI Model Monitoring on the endpoint to detect feature drift and skew automatically
- **Batch prediction** — add a scheduled Vertex AI Batch Prediction job for scoring large customer lists overnight
- **Alerting** — Cloud Monitoring alerts when endpoint latency spikes or prediction volume drops unexpectedly
