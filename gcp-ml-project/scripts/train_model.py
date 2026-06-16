"""Fetch features from BigQuery, train a churn model, save to GCS."""

from google.cloud import bigquery, storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, os

PROJECT_ID = "serious-bliss-256222"
BUCKET     = "serious-bliss-256222-ml-data"
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
      COUNT(t.transaction_id) AS total_transactions,
      COALESCE(SUM(t.amount), 0)  AS total_spend,
      COALESCE(AVG(t.amount), 0)  AS avg_transaction,
      c.churned                   AS label
    FROM {DATASET_ID}.customers c
    LEFT JOIN {DATASET_ID}.transactions t USING (customer_id)
    GROUP BY 1,2,3,4,5,6,10
    """
    client.query(sql).result()
    print("  ✓ Features table created in BigQuery")


def fetch_features():
    client = bigquery.Client(project=PROJECT_ID)
    return client.query(f"SELECT * FROM {DATASET_ID}.features").to_dataframe()


def train_and_upload(df):
    feature_cols = ["age", "tenure_months", "monthly_charge", "num_products",
                    "support_calls", "total_transactions", "total_spend", "avg_transaction"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\n📊 Model Performance:")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    # Save and upload
    os.makedirs("/tmp/churn_model", exist_ok=True)
    model_path = "/tmp/churn_model/model.joblib"
    joblib.dump(model, model_path)

    client = storage.Client(project=PROJECT_ID)
    client.bucket(BUCKET).blob("models/churn/model.joblib").upload_from_filename(model_path)
    print(f"  ✓ Model saved → gs://{BUCKET}/models/churn/model.joblib")


if __name__ == "__main__":
    print("\n🔧 Creating features table...")
    create_features_table()

    print("📥 Fetching features...")
    df = fetch_features()
    print(f"  ✓ {len(df)} rows loaded")

    print("\n🤖 Training model...")
    train_and_upload(df)
    print("\n✅ Done!")
