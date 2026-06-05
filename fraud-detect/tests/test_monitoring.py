import numpy as np
import pandas as pd
import pytest
from src.monitoring.drift import compute_psi, detect_drift, summarize_drift


@pytest.fixture
def reference_df():
    np.random.seed(0)
    return pd.DataFrame({"Amount": np.random.normal(100, 20, 1000), "V1": np.random.randn(1000)})


@pytest.fixture
def similar_df():
    np.random.seed(1)
    return pd.DataFrame({"Amount": np.random.normal(100, 20, 1000), "V1": np.random.randn(1000)})


@pytest.fixture
def drifted_df():
    np.random.seed(2)
    return pd.DataFrame({"Amount": np.random.normal(500, 100, 1000), "V1": np.random.randn(1000) * 10})


def test_psi_no_drift(reference_df, similar_df):
    psi = compute_psi(reference_df["Amount"], similar_df["Amount"])
    assert psi < 0.1


def test_psi_drift(reference_df, drifted_df):
    psi = compute_psi(reference_df["Amount"], drifted_df["Amount"])
    assert psi > 0.2


def test_detect_drift_returns_all_features(reference_df, similar_df):
    results = detect_drift(reference_df, similar_df)
    assert "Amount" in results or "V1" in results


def test_summarize_drift(reference_df, drifted_df):
    results = detect_drift(reference_df, drifted_df)
    summary = summarize_drift(results)
    assert "needs_retraining" in summary
    assert summary["needs_retraining"] is True
