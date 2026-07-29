"""
app.py

Standalone entry point. Run with:

    streamlit run app.py

Two tabs:
  1. Directory  — search ECW, generate/download a QR per patient
  2. Scan       — scan/paste a QR payload, display that patient's
                   info read-only, fetched live from ECW

Nothing in this app writes to ECW. There is no save/submit path
that touches the FHIR API beyond GET requests.
"""

from __future__ import annotations

import streamlit as st

import config
from ecw_client import ECWClient
from patient_directory_view import render_patient_directory
from patient_lookup_view import render_patient_lookup


st.set_page_config(page_title="ECW Patient QR", layout="wide")
st.title("Patient QR — ECW Lookup (Read Only)")

client = ECWClient(
    base_url=config.ECW_BASE_URL,
    client_id=config.ECW_CLIENT_ID,
    client_secret=config.ECW_CLIENT_SECRET,
    token_url=config.ECW_TOKEN_URL,
)

tab_directory, tab_scan = st.tabs(["Directory", "Scan"])

with tab_directory:
    render_patient_directory(client)

with tab_scan:
    render_patient_lookup(client)
