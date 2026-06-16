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
