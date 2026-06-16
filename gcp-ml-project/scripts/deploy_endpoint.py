"""Deploy the latest registered Vertex AI model to a prediction endpoint."""

from google.cloud import aiplatform

PROJECT_ID    = "serious-bliss-256222"
REGION        = "us-central1"
MODEL_NAME    = "churn-prediction-model"
ENDPOINT_NAME = "churn-prediction-endpoint"


def get_latest_model():
    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        raise RuntimeError(
            f"No model found with display name '{MODEL_NAME}'. "
            "Run train_vertex.py first."
        )
    model = models[0]
    print(f"  ✓ Model found: {model.resource_name}")
    return model


def get_or_create_endpoint():
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_NAME}"',
        order_by="create_time desc",
    )
    if endpoints:
        endpoint = endpoints[0]
        print(f"  ✓ Using existing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
        print(f"  ✓ Created endpoint: {endpoint.resource_name}")
    return endpoint


def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"\n📦 Fetching latest model '{MODEL_NAME}'...")
    model = get_latest_model()

    print(f"\n🔌 Preparing endpoint '{ENDPOINT_NAME}'...")
    endpoint = get_or_create_endpoint()

    print(f"\n🚀 Deploying model to endpoint (this takes ~5-10 min)...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=MODEL_NAME,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        sync=True,
    )

    print(f"\n✅ Deployment complete!")
    print(f"   Endpoint resource : {endpoint.resource_name}")
    print(f"\n📡 Test with:")
    print(f"""
   from google.cloud import aiplatform
   endpoint = aiplatform.Endpoint('{endpoint.resource_name}')
   prediction = endpoint.predict(instances=[[34, 24, 65.5, 2, 1, 5, 300.0, 60.0]])
   print(prediction)
""")


if __name__ == "__main__":
    main()
