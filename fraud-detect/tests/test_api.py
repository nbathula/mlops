import pytest
import json
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def client():
    with patch("api.model_loader.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_load.return_value = mock_model

        from api.app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


@pytest.fixture
def transaction_payload():
    return {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 150.0,
        "Time": 50000.0,
    }


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_returns_prediction(client, transaction_payload):
    resp = client.post(
        "/predict",
        data=json.dumps(transaction_payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data


def test_predict_invalid_payload(client):
    resp = client.post(
        "/predict",
        data=json.dumps({"Amount": -10}),
        content_type="application/json",
    )
    assert resp.status_code == 400
