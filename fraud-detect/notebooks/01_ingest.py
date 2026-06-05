# Databricks notebook source

# COMMAND ----------
dbutils.widgets.text("s3_bucket", "fraud-detect-artifacts")
s3_bucket = dbutils.widgets.get("s3_bucket")

csv_s3_path = f"s3a://{s3_bucket}/raw/creditcard.csv"
delta_path  = f"s3a://{s3_bucket}/delta/transactions"

# COMMAND ----------
aws_key    = dbutils.secrets.get(scope="fraud-detect", key="aws_access_key_id")
aws_secret = dbutils.secrets.get(scope="fraud-detect", key="aws_secret_access_key")

spark.conf.set("fs.s3a.access.key",  aws_key)
spark.conf.set("fs.s3a.secret.key",  aws_secret)
spark.conf.set("fs.s3a.endpoint",    "s3.amazonaws.com")
spark.conf.set("fs.s3a.impl",        "org.apache.hadoop.fs.s3a.S3AFileSystem")

# COMMAND ----------
print(f"Reading CSV from {csv_s3_path}")
df = spark.read.csv(csv_s3_path, header=True, inferSchema=True)
print(f"Row count: {df.count()}")

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(delta_path)
print(f"Delta table written to {delta_path}")

dbutils.jobs.taskValues.set(key="delta_path", value=delta_path)
