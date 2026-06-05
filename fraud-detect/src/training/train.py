import os
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from loguru import logger
from dotenv import load_dotenv

from src.ingestion.ingest import load_sample_data
from src.features.feature_engineering import build_features, split_features_labels

load_dotenv()


def train(data_path: str = "data/creditcard.csv"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))

    df = load_sample_data(data_path)
    df = build_features(df)
    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE — train size: {X_train_res.shape[0]}")

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": 1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision_fraud", report["1"]["precision"])
        mlflow.log_metric("recall_fraud", report["1"]["recall"])
        mlflow.log_metric("f1_fraud", report["1"]["f1-score"])

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=os.getenv("MODEL_NAME", "fraud-detection-model"),
        )

        logger.info(f"ROC-AUC: {auc:.4f}")
        logger.info(f"Fraud Precision: {report['1']['precision']:.4f}")
        logger.info(f"Fraud Recall: {report['1']['recall']:.4f}")

    return model


if __name__ == "__main__":
    train()
