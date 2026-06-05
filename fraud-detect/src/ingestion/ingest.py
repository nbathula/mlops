import os
import boto3
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession


def get_spark():
    return SparkSession.builder.appName("fraud-detect-ingestion").getOrCreate()


def download_from_s3(bucket: str, key: str, local_path: str):
    s3 = boto3.client("s3")
    logger.info(f"Downloading s3://{bucket}/{key} -> {local_path}")
    s3.download_file(bucket, key, local_path)


def ingest_raw_to_delta(csv_path: str, delta_table_path: str):
    spark = get_spark()
    logger.info(f"Ingesting {csv_path} into Delta table at {delta_table_path}")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df.write.format("delta").mode("overwrite").save(delta_table_path)
    logger.info(f"Ingested {df.count()} rows")
    return df


def load_sample_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load data locally for development without Spark."""
    logger.info(f"Loading sample data from {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    bucket = os.getenv("S3_BUCKET", "fraud-detect-artifacts")
    ingest_raw_to_delta(
        csv_path=f"s3a://{bucket}/raw/creditcard.csv",
        delta_table_path=f"s3a://{bucket}/delta/transactions",
    )
