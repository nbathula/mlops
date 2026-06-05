import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index — detects feature distribution shift."""
    expected_perc = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=buckets)[0] / len(actual)

    # Avoid division by zero
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run KS test on all numeric features and PSI on Amount."""
    results = {}
    numeric_cols = reference.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        ks_stat, p_value = stats.ks_2samp(reference[col], current[col])
        drifted = p_value < 0.05
        results[col] = {"ks_stat": ks_stat, "p_value": p_value, "drifted": drifted}
        if drifted:
            logger.warning(f"Drift detected on feature '{col}' (p={p_value:.4f})")

    if "Amount" in reference.columns:
        psi = compute_psi(reference["Amount"], current["Amount"])
        results["amount_psi"] = {"psi": psi, "drifted": psi > 0.2}
        logger.info(f"Amount PSI: {psi:.4f}")

    return results


def summarize_drift(drift_results: dict) -> dict:
    drifted_features = [k for k, v in drift_results.items() if v.get("drifted")]
    return {
        "total_features": len(drift_results),
        "drifted_count": len(drifted_features),
        "drifted_features": drifted_features,
        "needs_retraining": len(drifted_features) > 0,
    }
