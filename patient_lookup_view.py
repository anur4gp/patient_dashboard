"""
patient_lookup_view.py

Streamlit rendering for the Scan tab, wired to ECW instead of the
Excel backend for this specific flow. Read-only: there is no save
button, no write call anywhere in this file, by design.

Drop-in usage inside app.py's Scan tab:

    from ecw_integration.ecw_client import ECWClient, ECWAPIError
    from ecw_integration.patient_lookup_view import render_patient_lookup

    ecw_client = ECWClient(
        base_url=config.ECW_BASE_URL,
        client_id=config.ECW_CLIENT_ID,
        client_secret=config.ECW_CLIENT_SECRET,
        token_url=config.ECW_TOKEN_URL,
    )

    with tab_scan:
        render_patient_lookup(ecw_client)
"""

from __future__ import annotations

import streamlit as st

from ecw_client import ECWClient, ECWAPIError, ECWAuthError, PatientRecord
from patient_qr_service import decode_scanned_payload


def render_patient_lookup(client: ECWClient) -> None:
    st.subheader("Patient Lookup (ECW — read only)")

    scanned_text = st.text_input(
        "Scan QR code",
        placeholder="Scan or paste QR payload here",
        key="ecw_qr_scan_input",
    )

    if not scanned_text:
        st.caption("Waiting for a scan…")
        return

    try:
        patient_id = decode_scanned_payload(scanned_text)
    except ValueError as exc:
        st.error(f"Could not read that QR code: {exc}")
        return

    try:
        with st.spinner("Fetching patient record from ECW…"):
            patient = client.get_patient_by_id(patient_id)
    except ECWAuthError:
        st.error("Could not authenticate with ECW. Check credentials/config.")
        return
    except ECWAPIError as exc:
        if exc.status_code == 404:
            st.warning("No matching patient found in ECW for this QR code.")
        elif exc.status_code == 403:
            st.error("Access denied — this app's ECW scope does not permit Patient reads.")
        elif exc.status_code == 429:
            st.error("ECW rate limit hit — please wait a moment and rescan.")
        else:
            st.error(f"ECW API error ({exc.status_code}). Please try again.")
        return

    _render_patient_card(patient)


def _render_patient_card(patient: PatientRecord) -> None:
    st.success(f"Found: {patient.full_name}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Demographics**")
        st.write(f"MRN: {patient.mrn or '—'}")
        st.write(f"DOB: {patient.birth_date or '—'}")
        st.write(f"Gender: {patient.gender or '—'}")
        st.write(f"Active: {'Yes' if patient.active else 'No'}")

    with col2:
        st.markdown("**Contact**")
        st.write(f"Phone (home): {patient.phone_home or '—'}")
        st.write(f"Phone (mobile): {patient.phone_mobile or '—'}")
        st.write(f"Email: {patient.email or '—'}")

    st.markdown("**Address**")
    address_parts = [
        patient.address_line,
        patient.address_city,
        patient.address_state,
        patient.address_postal_code,
    ]
    st.write(", ".join(p for p in address_parts if p) or "—")

    st.caption(f"ECW record last updated: {patient.last_updated or 'unknown'}")

    with st.expander("Raw FHIR resource (debug)"):
        st.json(patient.raw)
