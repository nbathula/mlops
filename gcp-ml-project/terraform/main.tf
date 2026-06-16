terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Enable required GCP APIs ──────────────────────────────────────────────────
resource "google_project_service" "apis" {
  for_each = toset([
    "storage.googleapis.com",
    "bigquery.googleapis.com",
    "aiplatform.googleapis.com",
    "compute.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

# ── Cloud Storage bucket ───────────────────────────────────────────────────────
resource "google_storage_bucket" "ml_data" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis]
}

# Folder placeholders inside the bucket
resource "google_storage_bucket_object" "folders" {
  for_each = toset(["raw/", "processed/", "models/"])
  name     = each.key
  bucket   = google_storage_bucket.ml_data.name
  content  = " "
}

# ── BigQuery dataset ───────────────────────────────────────────────────────────
resource "google_bigquery_dataset" "ml_dataset" {
  dataset_id  = "ml_churn"
  description = "Customer churn ML dataset"
  location    = var.region

  depends_on = [google_project_service.apis]
}

# Customers table
resource "google_bigquery_table" "customers" {
  dataset_id          = google_bigquery_dataset.ml_dataset.dataset_id
  table_id            = "customers"
  deletion_protection = false

  schema = jsonencode([
    { name = "customer_id",    type = "STRING",  mode = "REQUIRED" },
    { name = "age",            type = "INTEGER", mode = "NULLABLE" },
    { name = "tenure_months",  type = "INTEGER", mode = "NULLABLE" },
    { name = "monthly_charge", type = "FLOAT",   mode = "NULLABLE" },
    { name = "num_products",   type = "INTEGER", mode = "NULLABLE" },
    { name = "support_calls",  type = "INTEGER", mode = "NULLABLE" },
    { name = "churned",        type = "INTEGER", mode = "NULLABLE" }
  ])
}

# Transactions table
resource "google_bigquery_table" "transactions" {
  dataset_id          = google_bigquery_dataset.ml_dataset.dataset_id
  table_id            = "transactions"
  deletion_protection = false

  schema = jsonencode([
    { name = "transaction_id",  type = "STRING", mode = "REQUIRED" },
    { name = "customer_id",     type = "STRING", mode = "NULLABLE" },
    { name = "date",            type = "DATE",   mode = "NULLABLE" },
    { name = "amount",          type = "FLOAT",  mode = "NULLABLE" },
    { name = "category",        type = "STRING", mode = "NULLABLE" },
    { name = "payment_method",  type = "STRING", mode = "NULLABLE" }
  ])
}

# ── Service Account for ML workloads ──────────────────────────────────────────
resource "google_service_account" "ml_sa" {
  account_id   = "ml-workload-sa"
  display_name = "ML Workload Service Account"
}

resource "google_project_iam_member" "ml_sa_roles" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/aiplatform.user",
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.ml_sa.email}"
}
