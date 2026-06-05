# Databricks notebook source

# COMMAND ----------
delta_path    = dbutils.jobs.taskValues.get(taskKey="ingest_data", key="delta_path")
features_path = delta_path.replace("delta/transactions", "delta/features")

aws_key    = dbutils.secrets.get(scope="fraud-detect", key="aws_access_key_id")
aws_secret = dbutils.secrets.get(scope="fraud-detect", key="aws_secret_access_key")

spark.conf.set("fs.s3a.access.key", aws_key)
spark.conf.set("fs.s3a.secret.key", aws_secret)

# COMMAND ----------
from pyspark.sql import functions as F

df = spark.read.format("delta").load(delta_path)
print(f"Loaded {df.count()} rows")

# COMMAND ----------
stats        = df.select(F.mean("Amount").alias("mean"), F.stddev("Amount").alias("std")).first()
amount_mean  = stats["mean"]
amount_std   = stats["std"]

df = (
    df
    .withColumn("amount_log",    F.log1p(F.col("Amount")))
    .withColumn("amount_zscore", (F.col("Amount") - amount_mean) / amount_std)
    .withColumn("hour_of_day",   ((F.col("Time") % 86400) / 3600).cast("int"))
    .withColumn("is_night",      F.when(
        (F.col("hour_of_day") < 6) | (F.col("hour_of_day") >= 22), 1
    ).otherwise(0))
    .drop("Time", "Amount")
)

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(features_path)
print(f"Feature table written to {features_path} — {df.count()} rows, {len(df.columns)} columns")

dbutils.jobs.taskValues.set(key="features_path", value=features_path)
