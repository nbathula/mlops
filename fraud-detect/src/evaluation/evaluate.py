import os
import mlflow
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from loguru import logger
from dotenv import load_dotenv

from src.ingestion.ingest import load_sample_data
from src.features.feature_engineering import build_features, split_features_labels

load_dotenv()

PROMOTION_THRESHOLD_AUC = 0.95


def load_model_from_registry(model_name: str, stage: str = "Staging"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading model from {model_uri}")
    return mlflow.xgboost.load_model(model_uri)


def evaluate(data_path: str = "data/creditcard.csv", stage: str = "Staging") -> dict:
    model_name = os.getenv("MODEL_NAME", "fraud-detection-model")
    model = load_model_from_registry(model_name, stage)

    df = load_sample_data(data_path)
    df = build_features(df)
    X, y = split_features_labels(df)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y, y_prob),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def should_promote(metrics: dict) -> bool:
    return metrics["roc_auc"] >= PROMOTION_THRESHOLD_AUC


if __name__ == "__main__":
    metrics = evaluate()
    if should_promote(metrics):
        logger.info("Model meets promotion threshold — promoting to Production")
    else:
        logger.warning("Model does NOT meet threshold — not promoting")
