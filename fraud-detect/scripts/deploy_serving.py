"""
Creates (or updates) a Databricks Model Serving endpoint for the fraud-detection model.
Run after the training job has promoted a model version to Production.

Usage:
    python scripts/deploy_serving.py
"""

import os
import time
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

load_dotenv()

ENDPOINT_NAME = "fraud-detect-endpoint"
MODEL_NAME    = os.getenv("MODEL_NAME", "fraud-detection-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")  # override after training


def main():
    client = WorkspaceClient(
        host  = os.environ["DATABRICKS_HOST"],
        token = os.environ["DATABRICKS_TOKEN"],
    )

    served_entity = ServedEntityInput(
        entity_name           = MODEL_NAME,
        entity_version        = MODEL_VERSION,
        workload_size         = "Small",
        scale_to_zero_enabled = True,
    )

    config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity])

    existing = {e.name: e for e in client.serving_endpoints.list()}

    if ENDPOINT_NAME in existing:
        print(f"Updating existing endpoint '{ENDPOINT_NAME}'...")
        client.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served_entity])
    else:
        print(f"Creating endpoint '{ENDPOINT_NAME}'...")
        client.serving_endpoints.create(name=ENDPOINT_NAME, config=config)

    # Wait until ready
    print("Waiting for endpoint to become ready...")
    for _ in range(60):
        endpoint = client.serving_endpoints.get(ENDPOINT_NAME)
        state = endpoint.state.ready if endpoint.state else None
        print(f"  state: {state}")
        if str(state) == "READY":
            break
        time.sleep(15)

    endpoint = client.serving_endpoints.get(ENDPOINT_NAME)
    host     = os.environ["DATABRICKS_HOST"].rstrip("/")
    url      = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    print(f"\nEndpoint ready: {url}")
    print("Example request:")
    print(f"""
  curl -X POST {url} \\
    -H 'Authorization: Bearer $DATABRICKS_TOKEN' \\
    -H 'Content-Type: application/json' \\
    -d '{{"dataframe_split": {{"columns": ["V1","V2","...","amount_log","amount_zscore","hour_of_day","is_night"], "data": [[...]]}}}}'
""")


if __name__ == "__main__":
    main()
