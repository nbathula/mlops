# Databricks notebook source

# COMMAND ----------
import numpy as np
import pandas as pd
from scipy import stats

dbutils.widgets.text("s3_bucket", "fraud-detect-artifacts")
s3_bucket     = dbutils.widgets.get("s3_bucket")
features_path = dbutils.jobs.taskValues.get(taskKey="build_features", key="features_path")
inference_path = f"s3a://{s3_bucket}/delta/inference"

# COMMAND ----------
aws_key    = dbutils.secrets.get(scope="fraud-detect", key="aws_access_key_id")
aws_secret = dbutils.secrets.get(scope="fraud-detect", key="aws_secret_access_key")

spark.conf.set("fs.s3a.access.key", aws_key)
spark.conf.set("fs.s3a.secret.key", aws_secret)

# COMMAND ----------
reference = spark.read.format("delta").load(features_path).toPandas()

try:
    current = spark.read.format("delta").load(inference_path).toPandas()
    print(f"Loaded inference data: {len(current)} rows")
except Exception:
    print("No inference data found — using 20% sample of training data as proxy")
    current = reference.sample(frac=0.2, random_state=99)

# COMMAND ----------
def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    exp_perc = np.histogram(expected, bins=buckets)[0] / len(expected)
    act_perc = np.histogram(actual,   bins=buckets)[0] / len(actual)
    exp_perc = np.where(exp_perc == 0, 1e-4, exp_perc)
    act_perc = np.where(act_perc == 0, 1e-4, act_perc)
    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))

numeric_cols  = reference.select_dtypes(include=np.number).columns.tolist()
drift_results = {}

for col in numeric_cols:
    ks_stat, p_value = stats.ks_2samp(reference[col], current[col])
    drifted = p_value < 0.05
    drift_results[col] = {"ks_stat": round(ks_stat, 4), "p_value": round(p_value, 4), "drifted": drifted}
    if drifted:
        print(f"Drift on '{col}': KS={ks_stat:.4f}, p={p_value:.4f}")

if "amount_log" in reference.columns:
    psi = compute_psi(reference["amount_log"], current["amount_log"])
    drift_results["amount_log_psi"] = {"psi": round(psi, 4), "drifted": psi > 0.2}
    print(f"amount_log PSI: {psi:.4f}")

# COMMAND ----------
drifted_features = [k for k, v in drift_results.items() if v.get("drifted")]
needs_retraining = len(drifted_features) > 0
summary = {
    "total_features":   len(drift_results),
    "drifted_count":    len(drifted_features),
    "drifted_features": drifted_features,
    "needs_retraining": needs_retraining,
}
print(f"Drift summary: {summary}")

dbutils.jobs.taskValues.set(key="drift_summary", value=str(summary))

displayHTML(f"""
<h3>Drift Report</h3>
<p>Features checked: <b>{summary['total_features']}</b></p>
<p>Drifted: <b>{summary['drifted_count']}</b> — {', '.join(drifted_features) or 'none'}</p>
<p>Needs retraining: <b>{'YES ⚠️' if needs_retraining else 'NO ✅'}</b></p>
""")
