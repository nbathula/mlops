import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Monitoring", layout="wide")
st.title("📊 Model Monitoring")
st.markdown("Track model performance and data drift over time.")

# Simulated metrics — replace with real MLflow queries in production
np.random.seed(42)
dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
auc_scores = np.clip(0.97 + np.random.normal(0, 0.01, 30), 0.90, 1.0)
fraud_rates = np.clip(0.002 + np.random.normal(0, 0.0005, 30), 0, 0.01)
drift_scores = np.clip(np.random.normal(0.05, 0.03, 30), 0, 0.3)

metrics_df = pd.DataFrame({
    "date": dates,
    "roc_auc": auc_scores,
    "fraud_rate": fraud_rates,
    "drift_psi": drift_scores,
})

st.subheader("Model Performance (Last 30 Days)")
col1, col2, col3 = st.columns(3)
col1.metric("Current ROC-AUC", f"{auc_scores[-1]:.4f}", delta=f"{auc_scores[-1]-auc_scores[-2]:+.4f}")
col2.metric("Fraud Rate", f"{fraud_rates[-1]:.3%}", delta=f"{fraud_rates[-1]-fraud_rates[-2]:+.4%}")
col3.metric("Drift PSI", f"{drift_scores[-1]:.3f}", delta=f"{drift_scores[-1]-drift_scores[-2]:+.3f}")

fig_auc = px.line(metrics_df, x="date", y="roc_auc", title="ROC-AUC Over Time")
fig_auc.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Threshold")
st.plotly_chart(fig_auc, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig_fraud = px.bar(metrics_df, x="date", y="fraud_rate", title="Daily Fraud Rate")
    st.plotly_chart(fig_fraud, use_container_width=True)

with col2:
    fig_drift = px.line(metrics_df, x="date", y="drift_psi", title="Data Drift (PSI)")
    fig_drift.add_hline(y=0.2, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig_drift.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Critical")
    st.plotly_chart(fig_drift, use_container_width=True)

st.subheader("Drift Status by Feature")
features = [f"V{i}" for i in range(1, 10)] + ["amount_log", "amount_zscore", "hour_of_day"]
drift_status = pd.DataFrame({
    "Feature": features,
    "PSI": np.clip(np.random.normal(0.05, 0.06, len(features)), 0, 0.35),
    "KS p-value": np.clip(np.random.uniform(0.01, 0.5, len(features)), 0, 1),
})
drift_status["Drifted"] = drift_status["PSI"] > 0.2

st.dataframe(
    drift_status.style.applymap(
        lambda v: "background-color: #ffcccc" if v else "",
        subset=["Drifted"]
    ),
    use_container_width=True,
)
