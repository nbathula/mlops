#!/usr/bin/env bash
# Creates the Databricks secret scope and loads AWS credentials into it.
# Run once before deploying the bundle.
# Usage: bash scripts/setup_secrets.sh

set -euo pipefail

SCOPE="fraud-detect"

echo "Creating secret scope '$SCOPE' (ignores error if it already exists)..."
databricks secrets create-scope "$SCOPE" 2>/dev/null || true

echo "Writing AWS credentials..."
databricks secrets put-secret "$SCOPE" aws_access_key_id     --string-value "$AWS_ACCESS_KEY_ID"
databricks secrets put-secret "$SCOPE" aws_secret_access_key --string-value "$AWS_SECRET_ACCESS_KEY"

echo "Done. Secrets in scope '$SCOPE':"
databricks secrets list-secrets "$SCOPE"
