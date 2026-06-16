"""Upload CSV files to GCS and load them into BigQuery."""

from google.cloud import storage, bigquery
import os

PROJECT_ID  = "serious-bliss-256222"
BUCKET_NAME = "serious-bliss-256222-ml-data"
DATASET_ID  = "ml_churn"
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")


def upload_to_gcs(local_file: str, gcs_path: str):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_file)
    print(f"  ✓ Uploaded {os.path.basename(local_file)} → gs://{BUCKET_NAME}/{gcs_path}")


def load_csv_to_bq(gcs_uri: str, table_id: str):
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=False,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
    job.result()
    table = client.get_table(table_ref)
    print(f"  ✓ Loaded {table.num_rows} rows → {table_ref}")


if __name__ == "__main__":
    print("\n📤 Uploading CSVs to GCS...")
    upload_to_gcs(os.path.join(DATA_DIR, "customers.csv"),    "raw/customers.csv")
    upload_to_gcs(os.path.join(DATA_DIR, "transactions.csv"), "raw/transactions.csv")

    print("\n📊 Loading into BigQuery...")
    load_csv_to_bq(f"gs://{BUCKET_NAME}/raw/customers.csv",    "customers")
    load_csv_to_bq(f"gs://{BUCKET_NAME}/raw/transactions.csv", "transactions")

    print("\n✅ Done! Open BigQuery console to explore your data.")
    print(f"   https://console.cloud.google.com/bigquery?project={PROJECT_ID}")
