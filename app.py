"""
FemScan-AI — Streamlit entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="FemScan-AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FemScan-AI")
st.caption("AI-assisted cervical cell classification & endometriosis risk scoring")

st.info("Dashboard UI coming soon. See `docs/OUTPUT_CONTRACTS.md` for integration specs.")
