"""
patient_directory_view.py

Search ECW for a patient (by name/DOB/etc, per the query parameter
combinations ECW supports) and generate a printable QR code for
whichever one you're onboarding. This is the "front half" of the
flow — patient_lookup_view.py is the "back half" (scan the QR you
made here, look the patient back up, display their info).

Still read-only against ECW: search + display only. The QR image
itself is generated locally, not written back to ECW in any way.
"""

from __future__ import annotations

import streamlit as st

from ecw_client import ECWClient, ECWAPIError, ECWAuthError
from patient_qr_service import generate_patient_qr


def render_patient_directory(client: ECWClient) -> None:
    st.subheader("Find Patient & Generate QR (ECW — read only)")

    with st.form("ecw_patient_search"):
        col1, col2 = st.columns(2)
        with col1:
            family = st.text_input("Last name")
            birthdate = st.text_input("Date of birth (yyyy-mm-dd)")
        with col2:
            given = st.text_input("First name")
            gender = st.selectbox("Gender", ["", "male", "female", "unknown"])
        submitted = st.form_submit_button("Search")

    if not submitted:
        return

    if not (family or given or birthdate):
        st.warning("Enter at least a last name, first name, or date of birth.")
        return

    try:
        with st.spinner("Searching ECW…"):
            results = client.search_patient(
                family=family or None,
                given=given or None,
                birthdate=birthdate or None,
                gender=gender or None,
            )
    except ECWAuthError:
        st.error("Could not authenticate with ECW. Check credentials/config.")
        return
    except ECWAPIError as exc:
        if exc.status_code == 403:
            st.error("Access denied — this app's ECW scope does not permit Patient reads.")
        elif exc.status_code == 429:
            st.error("ECW rate limit hit — please wait a moment and search again.")
        else:
            st.error(f"ECW API error ({exc.status_code}). Please try again.")
        return

    if not results:
        st.info("No matching patients found.")
        return

    st.write(f"Found {len(results)} matching patient(s):")

    for patient in results:
        with st.expander(f"{patient.full_name} — DOB {patient.birth_date or 'unknown'}"):
            st.write(f"MRN: {patient.mrn or '—'}")
            st.write(f"Gender: {patient.gender or '—'}")
            st.write(f"Phone: {patient.phone_home or patient.phone_mobile or '—'}")

            qr_png = generate_patient_qr(patient.fhir_id)
            st.image(qr_png, caption=f"QR for {patient.full_name}", width=200)
            st.download_button(
                label="Download QR (PNG)",
                data=qr_png,
                file_name=f"patient_qr_{patient.mrn or patient.fhir_id}.png",
                mime="image/png",
                key=f"dl_{patient.fhir_id}",
            )
