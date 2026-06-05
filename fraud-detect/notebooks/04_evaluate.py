# Databricks notebook source

# COMMAND ----------
import mlflow
import mlflow.xgboost
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

dbutils.widgets.text("model_name", "fraud-detection-model")
model_name    = dbutils.widgets.get("model_name")
features_path = dbutils.jobs.taskValues.get(taskKey="build_features", key="features_path")

PROMOTION_THRESHOLD_AUC = 0.95

# COMMAND ----------
aws_key    = dbutils.secrets.get(scope="fraud-detect", key="aws_access_key_id")
aws_secret = dbutils.secrets.get(scope="fraud-detect", key="aws_secret_access_key")

spark.conf.set("fs.s3a.access.key", aws_key)
spark.conf.set("fs.s3a.secret.key", aws_secret)

mlflow.set_tracking_uri("databricks")
client = mlflow.tracking.MlflowClient()

# COMMAND ----------
# Use latest registered version (just trained)
all_versions = client.search_model_versions(f"name='{model_name}'")
version = sorted(all_versions, key=lambda v: int(v.version))[-1]
print(f"Evaluating version {version.version} (current stage: {version.current_stage})")

model = mlflow.xgboost.load_model(f"models:/{model_name}/{version.version}")

# COMMAND ----------
df = spark.read.format("delta").load(features_path).toPandas()
feature_cols = [c for c in df.columns if c != "Class"]
X, y = df[feature_cols], df["Class"]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

metrics = {
    "roc_auc":   round(roc_auc_score(y, y_prob), 4),
    "precision": round(precision_score(y, y_pred), 4),
    "recall":    round(recall_score(y, y_pred), 4),
    "f1":        round(f1_score(y, y_pred), 4),
}
print(f"Metrics: {metrics}")

# COMMAND ----------
if metrics["roc_auc"] >= PROMOTION_THRESHOLD_AUC:
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Version {version.version} promoted to Production (AUC={metrics['roc_auc']})")
    dbutils.jobs.taskValues.set(key="promoted",      value=True)
    dbutils.jobs.taskValues.set(key="model_version", value=version.version)
else:
    print(f"AUC {metrics['roc_auc']} below threshold {PROMOTION_THRESHOLD_AUC} — not promoting")
    dbutils.jobs.taskValues.set(key="promoted",      value=False)
    dbutils.jobs.taskValues.set(key="model_version", value=version.version)
