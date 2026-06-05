import streamlit as st

st.set_page_config(
    page_title="Fraud Detect",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Fraud Detect")
st.markdown("Real-time credit card fraud detection powered by MLOps.")

st.sidebar.title("Navigation")
st.sidebar.markdown("Use the pages in the sidebar to navigate.")

col1, col2, col3 = st.columns(3)
col1.metric("Model Status", "Active")
col2.metric("Model Stage", "Production")
col3.metric("Threshold", "0.50")

st.markdown("---")
st.markdown("### Getting Started")
st.markdown("""
- **Predict** — Score a single transaction
- **Batch Score** — Upload a CSV and score in bulk
- **Monitoring** — View model performance and data drift
""")
