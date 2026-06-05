import pandas as pd
import numpy as np
from loguru import logger


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building features")
    df = df.copy()

    # Transaction amount features
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    # Time-based features (seconds from first transaction)
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)

    # Drop raw columns not needed for model
    df = df.drop(columns=["Time", "Amount"])

    logger.info(f"Feature matrix shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != "Class"]


def split_features_labels(df: pd.DataFrame):
    features = get_feature_columns(df)
    X = df[features]
    y = df["Class"]
    return X, y
