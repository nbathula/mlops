"""
Train a churn model using Vertex AI AutoML Tabular.
AutoML automatically selects the best algorithm and hyperparameters.
Run this as an alternative to train_vertex.py.
"""

from google.cloud import aiplatform

PROJECT_ID  = "serious-bliss-256222"
REGION      = "us-central1"
BUCKET      = "serious-bliss-256222-ml-data"
DATASET_ID  = "ml_churn"
BQ_SOURCE   = f"bq://{PROJECT_ID}.{DATASET_ID}.features"


def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

    # Register the BigQuery features table as a Vertex AI Dataset
    print("\n📂 Creating Vertex AI Tabular Dataset from BigQuery...")
    dataset = aiplatform.TabularDataset.create(
        display_name="churn-dataset",
        bq_source=BQ_SOURCE,
    )
    print(f"  ✓ Dataset: {dataset.resource_name}")

    # Run AutoML — tries dozens of algorithms, picks the best by AUC
    print("\n🤖 Launching AutoML Tabular training job...")
    print("   This typically takes 60-120 minutes for a full run.")
    print("   Reduce budget_milli_node_hours for faster (less optimal) results.\n")

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="churn-automl-training",
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-roc",
        column_transformations=[
            {"numeric": {"column_name": "age"}},
            {"numeric": {"column_name": "tenure_months"}},
            {"numeric": {"column_name": "monthly_charge"}},
            {"numeric": {"column_name": "num_products"}},
            {"numeric": {"column_name": "support_calls"}},
            {"numeric": {"column_name": "total_transactions"}},
            {"numeric": {"column_name": "total_spend"}},
            {"numeric": {"column_name": "avg_transaction"}},
        ],
    )

    model = job.run(
        dataset=dataset,
        target_column="label",
        budget_milli_node_hours=1000,      # 1 training hour — increase for better results
        model_display_name="churn-automl-model",
        disable_early_stopping=False,       # stop early if model converges
        sync=True,
    )

    print(f"\n✅ AutoML training complete!")
    print(f"   Model resource : {model.resource_name}")
    print(f"\n   Next → update deploy_endpoint.py MODEL_NAME to 'churn-automl-model'")
    print(f"   Then  → python scripts/deploy_endpoint.py")

    return model


if __name__ == "__main__":
    main()
