"""
FemScan AI — Streamlit Dashboard Entry Point
============================================
Person B builds this file.
All model outputs are defined in docs/OUTPUT_CONTRACTS.md.

Run:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="FemScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FemScan AI")
st.caption("AI-assisted cervical health screening — HPV risk · CIN detection · Endometriosis scoring")

st.info(
    "Dashboard UI — Person B builds here.  \n"
    "Model output shapes: see `docs/OUTPUT_CONTRACTS.md`  \n"
    "Demo assets: run `python scripts/prepare_demo.py` first."
)
