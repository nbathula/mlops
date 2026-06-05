# Databricks notebook source

# COMMAND ----------
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

dbutils.widgets.text("mlflow_experiment", "/fraud-detect/experiments/fraud-detection")
dbutils.widgets.text("model_name",        "fraud-detection-model")

mlflow_experiment = dbutils.widgets.get("mlflow_experiment")
model_name        = dbutils.widgets.get("model_name")
features_path     = dbutils.jobs.taskValues.get(taskKey="build_features", key="features_path")

# COMMAND ----------
aws_key    = dbutils.secrets.get(scope="fraud-detect", key="aws_access_key_id")
aws_secret = dbutils.secrets.get(scope="fraud-detect", key="aws_secret_access_key")

spark.conf.set("fs.s3a.access.key", aws_key)
spark.conf.set("fs.s3a.secret.key", aws_secret)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(mlflow_experiment)

# COMMAND ----------
df = spark.read.format("delta").load(features_path).toPandas()
print(f"Loaded {len(df)} rows for training")

feature_cols = [c for c in df.columns if c != "Class"]
X, y = df[feature_cols], df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — train size: {len(X_train_res)}")

# COMMAND ----------
params = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.05,
    "scale_pos_weight": 1,
    "eval_metric":      "logloss",
    "random_state":     42,
}

with mlflow.start_run() as run:
    mlflow.log_params(params)

    model = XGBClassifier(**params)
    model.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("roc_auc",         auc)
    mlflow.log_metric("precision_fraud", report["1"]["precision"])
    mlflow.log_metric("recall_fraud",    report["1"]["recall"])
    mlflow.log_metric("f1_fraud",        report["1"]["f1-score"])

    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        input_example=X_test.iloc[:5],
    )

    run_id = run.info.run_id
    print(f"Run {run_id} — ROC-AUC: {auc:.4f}")

dbutils.jobs.taskValues.set(key="run_id",     value=run_id)
dbutils.jobs.taskValues.set(key="model_name", value=model_name)
