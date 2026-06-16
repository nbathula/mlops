"""Training script that runs on Vertex AI managed compute."""

import os
import joblib
from google.cloud import bigquery
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd

PROJECT_ID = "serious-bliss-256222"
DATASET_ID = "ml_churn"


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


def train(df):
    feature_cols = [
        "age", "tenure_months", "monthly_charge", "num_products",
        "support_calls", "total_transactions", "total_spend", "avg_transaction",
    ]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n  ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    return model, {"roc_auc": round(auc, 4), "n_estimators": 100, "test_size": len(X_test)}


if __name__ == "__main__":
    # Vertex AI sets AIP_MODEL_DIR — save the model artifact there
    model_dir = os.environ.get("AIP_MODEL_DIR", "/tmp/churn_model")

    print("\n🔧 Creating features table in BigQuery...")
    create_features_table()

    print("\n📥 Fetching features...")
    df = fetch_features()

    print("\n🤖 Training RandomForest...")
    model, metrics = train(df)

    print(f"\n💾 Saving model to {model_dir} ...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    print(f"  ✓ model.joblib saved")

    print(f"\n📊 Metrics: {metrics}")
    print("\n✅ Training complete!")
