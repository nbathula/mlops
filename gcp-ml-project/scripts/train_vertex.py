"""
Train locally, log to Vertex AI Experiments, register model to Vertex AI Model Registry.
"""

import os
import joblib
import tempfile
from google.cloud import bigquery, storage, aiplatform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

PROJECT_ID    = "serious-bliss-256222"
REGION        = "us-central1"
BUCKET        = "serious-bliss-256222-ml-data"
EXPERIMENT    = "churn-prediction"
MODEL_NAME    = "churn-prediction-model"
DATASET_ID    = "ml_churn"
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"


# ── Feature engineering ────────────────────────────────────────────────────────

def create_features_table():
    client = bigquery.Client(project=PROJECT_ID)
    sql = f"""
    CREATE OR REPLACE TABLE {DATASET_ID}.features AS
    SELECT
      c.customer_id,
      c.age,
      c.tenure_months,
      c.monthly_charge,
      c.num_products,
      c.support_calls,
      COUNT(t.transaction_id)    AS total_transactions,
      COALESCE(SUM(t.amount), 0) AS total_spend,
      COALESCE(AVG(t.amount), 0) AS avg_transaction,
      c.churned                  AS label
    FROM {DATASET_ID}.customers c
    LEFT JOIN {DATASET_ID}.transactions t USING (customer_id)
    GROUP BY 1,2,3,4,5,6,10
    """
    client.query(sql).result()
    print("  ✓ Features table created in BigQuery")


def fetch_features():
    client = bigquery.Client(project=PROJECT_ID)
    df = client.query(f"SELECT * FROM {DATASET_ID}.features").to_dataframe()
    print(f"  ✓ {len(df)} rows fetched")
    return df


# ── Training ───────────────────────────────────────────────────────────────────

def train(df):
    feature_cols = [
        "age", "tenure_months", "monthly_charge", "num_products",
        "support_calls", "total_transactions", "total_spend", "avg_transaction",
    ]
    X, y = df[feature_cols], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n  ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    metrics = {
        "roc_auc": round(float(auc), 4),
        "n_estimators": 100,
        "test_rows": len(X_test),
    }
    return model, metrics


# ── Upload artifact to GCS ─────────────────────────────────────────────────────

def upload_model(model) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "model.joblib")
        joblib.dump(model, local_path)

        gcs_path = "models/churn/model.joblib"
        client = storage.Client(project=PROJECT_ID)
        client.bucket(BUCKET).blob(gcs_path).upload_from_filename(local_path)

    uri = f"gs://{BUCKET}/models/churn"
    print(f"  ✓ Model artifact uploaded → {uri}/model.joblib")
    return uri


# ── Register to Vertex AI Model Registry ──────────────────────────────────────

def register_model(artifact_uri: str) -> aiplatform.Model:
    model = aiplatform.Model.upload(
        display_name=MODEL_NAME,
        artifact_uri=artifact_uri,
        serving_container_image_uri=SERVING_IMAGE,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )
    print(f"  ✓ Model registered → {model.resource_name}")
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}",
        experiment=EXPERIMENT,
    )

    print("\n🔧 Creating features table in BigQuery...")
    create_features_table()

    print("\n📥 Fetching features...")
    df = fetch_features()

    print("\n🤖 Training RandomForest locally...")
    model, metrics = train(df)

    print("\n💾 Uploading model artifact to GCS...")
    artifact_uri = upload_model(model)

    print("\n📋 Registering model to Vertex AI Model Registry...")
    registered_model = register_model(artifact_uri)

    print("\n📊 Logging run to Vertex AI Experiments...")
    with aiplatform.start_run(run="run-01"):
        aiplatform.log_params({
            "n_estimators": metrics["n_estimators"],
            "framework": "sklearn-RandomForest",
            "training": "local",
        })
        aiplatform.log_metrics({
            "roc_auc": metrics["roc_auc"],
        })
    print("  ✓ Experiment run logged")

    print(f"\n✅ Done!")
    print(f"   Model registry : {registered_model.resource_name}")
    print(f"   Experiment     : {EXPERIMENT}")
    print(f"\n   Next → python scripts/deploy_endpoint.py")

    return registered_model


if __name__ == "__main__":
    main()
