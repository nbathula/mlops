import pandas as pd
import numpy as np
import pytest
from src.features.feature_engineering import build_features, split_features_labels


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n) for i in range(1, 29)},
        "Amount": np.abs(np.random.randn(n)) * 100,
        "Time": np.linspace(0, 172800, n),
        "Class": np.random.randint(0, 2, n),
    })
    return df


def test_build_features_adds_columns(sample_df):
    result = build_features(sample_df)
    assert "amount_log" in result.columns
    assert "amount_zscore" in result.columns
    assert "hour_of_day" in result.columns
    assert "is_night" in result.columns


def test_build_features_drops_raw(sample_df):
    result = build_features(sample_df)
    assert "Time" not in result.columns
    assert "Amount" not in result.columns


def test_split_features_labels(sample_df):
    df = build_features(sample_df)
    X, y = split_features_labels(df)
    assert "Class" not in X.columns
    assert len(X) == len(y)
