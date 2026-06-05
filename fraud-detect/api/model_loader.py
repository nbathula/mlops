import os
import mlflow.xgboost
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_model = None


def load_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        model_name = os.getenv("MODEL_NAME", "fraud-detection-model")
        stage = os.getenv("MODEL_STAGE", "Production")
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model: {model_uri}")
        _model = mlflow.xgboost.load_model(model_uri)
        logger.info("Model loaded successfully")
    return _model
