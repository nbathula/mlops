import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger
from dotenv import load_dotenv

from api.model_loader import load_model
from api.schemas import TransactionRequest, PredictionResponse
from src.features.feature_engineering import build_features

load_dotenv()

app = Flask(__name__)
CORS(app)


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.7:
        return "MEDIUM"
    return "HIGH"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": os.getenv("MODEL_NAME")}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = TransactionRequest(**request.get_json())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df = pd.DataFrame([payload.dict()])
    df = build_features(df)
    feature_cols = [c for c in df.columns if c != "Class"]

    model = load_model()
    prob = float(model.predict_proba(df[feature_cols])[0][1])
    is_fraud = prob >= 0.5

    response = PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        risk_level=get_risk_level(prob),
    )
    logger.info(f"Prediction: fraud={is_fraud}, prob={prob:.4f}")
    return jsonify(response.dict()), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()
        transactions = [TransactionRequest(**t) for t in data]
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df = pd.DataFrame([t.dict() for t in transactions])
    df = build_features(df)
    feature_cols = [c for c in df.columns if c != "Class"]

    model = load_model()
    probs = model.predict_proba(df[feature_cols])[:, 1]

    predictions = [
        {
            "is_fraud": bool(p >= 0.5),
            "fraud_probability": round(float(p), 4),
            "risk_level": get_risk_level(p),
        }
        for p in probs
    ]

    fraud_count = sum(1 for p in predictions if p["is_fraud"])
    return jsonify({
        "predictions": predictions,
        "total": len(predictions),
        "fraud_count": fraud_count,
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
