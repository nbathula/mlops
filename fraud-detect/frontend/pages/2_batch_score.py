import os
import io
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_BASE_URL", "http://localhost:5000")

st.set_page_config(page_title="Batch Score", layout="wide")
st.title("📂 Batch Score")
st.markdown("Upload a CSV of transactions and download scored results.")

st.markdown("**Expected columns:** `Time`, `Amount`, `V1` through `V28`")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("Score All Transactions", type="primary"):
        with st.spinner("Scoring..."):
            records = df.to_dict(orient="records")
            try:
                resp = requests.post(f"{API_URL}/predict/batch", json=records, timeout=60)
                result = resp.json()

                if resp.status_code == 200:
                    preds = pd.DataFrame(result["predictions"])
                    output = pd.concat([df.reset_index(drop=True), preds], axis=1)

                    st.success(f"Scored {result['total']} transactions — {result['fraud_count']} flagged as fraud")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Transactions", result["total"])
                    col2.metric("Fraud Detected", result["fraud_count"])
                    col3.metric("Fraud Rate", f"{result['fraud_count']/result['total']:.2%}")

                    st.dataframe(output)

                    csv = output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results CSV",
                        data=csv,
                        file_name="scored_transactions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(result.get("error"))
            except Exception as e:
                st.error(f"Could not reach API: {e}")
