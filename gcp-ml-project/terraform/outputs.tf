output "bucket_name" {
  description = "GCS bucket for ML data"
  value       = google_storage_bucket.ml_data.name
}

output "bigquery_dataset" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.ml_dataset.dataset_id
}

output "service_account_email" {
  description = "ML workload service account"
  value       = google_service_account.ml_sa.email
}

output "next_steps" {
  value = <<-EOT
    ✅ Infrastructure ready!

    Upload data:
      python scripts/upload_data.py

    BigQuery console:
      https://console.cloud.google.com/bigquery
  EOT
}
