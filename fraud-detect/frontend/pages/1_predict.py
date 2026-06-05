import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_BASE_URL", "http://localhost:5000")

st.set_page_config(page_title="Predict Transaction", layout="wide")
st.title("🔍 Predict Transaction")
st.markdown("Enter transaction details to check for fraud.")

with st.form("predict_form"):
    st.subheader("Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, value=150.0, step=0.01)
        time = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=50000.0)

    st.markdown("**PCA Features (V1–V28)**")
    cols = st.columns(4)
    v_values = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        v_values[f"V{i}"] = cols[col_idx].number_input(f"V{i}", value=0.0, format="%.4f")

    submitted = st.form_submit_button("Score Transaction", type="primary")

if submitted:
    payload = {"Amount": amount, "Time": time, **v_values}
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        result = resp.json()

        if resp.status_code == 200:
            risk = result["risk_level"]
            color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[risk]

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Detected", "YES" if result["is_fraud"] else "NO")
            col2.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
            col3.metric("Risk Level", risk)

            st.markdown(f"### :{color}[{risk} RISK]")
        else:
            st.error(f"API Error: {result.get('error')}")
    except Exception as e:
        st.error(f"Could not reach API: {e}")
