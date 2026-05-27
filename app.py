import streamlit as st
import pandas as pd
from services.excel_backend import ExcelBackend
from services.patient_service import PatientService
from services.qr_service import generate_qr
from services.scan_service import ScanService
from rapidfuzz import process, fuzz
from config import COMPARTMENTS

# init
backend = ExcelBackend("data/mock_patients.xlsx")
service = PatientService(backend)
scan_service = ScanService(backend)

# init load state session
if "df" not in st.session_state or "sheets" not in st.session_state:
    df, sheets = service.get_patients()
    st.session_state.df = df
    st.session_state.sheets = sheets


def refresh_state():
    df, sheets = service.get_patients()
    st.session_state.df = df
    st.session_state.sheets = sheets


# event logging
def log_event(patient_id, action, metadata=None):

    sheets = st.session_state.sheets

    sheets = scan_service.log_scan(
        sheets,
        patient_id,
        action=action,
        source="streamlit_ui",
        metadata=metadata
    )

    backend.write_patients(sheets)

    # refresh global state
    refresh_state()

    return st.session_state.df, st.session_state.sheets


# search function
def fuzzy_search_patients(df, query, limit=10):
    df = df.copy()

    df["full_name"] = (
        df["first_name"].astype(str).str.lower().str.strip()
        + " " +
        df["last_name"].astype(str).str.lower().str.strip()
    )

    results = process.extract(
        query.lower().strip(),
        df["full_name"].tolist(),
        scorer=fuzz.WRatio,
        limit=limit
    )

    matched = []
    for match, score, idx in results:
        if score > 50:
            matched.append(df.iloc[idx])

    return pd.DataFrame(matched)


st.title("Patient Dashboard")

# tabs
tab_scan, tab_directory, tab_logs = st.tabs([
    "🔍 Scan",
    "👤 Directory",
    "📜 Logs"
])

# scan tab
with tab_scan:

    st.subheader("Scan Patient")

    scan_input = st.text_input("Scan ID", key="scan_input")

    df = st.session_state.df
    sheets = st.session_state.sheets

    if scan_input:

        match = df[df["patient_uuid"] == scan_input]

        if len(match) > 0:
            st.session_state.active_patient = scan_input

            df, sheets = log_event(
                scan_input,
                action="SCAN_LOOKUP",
                metadata="scan_tab"
            )

        else:
            st.error("No patient found")

    active_id = st.session_state.get("active_patient")

    if active_id:

        df = st.session_state.df
        sheets = st.session_state.sheets

        patient = df[df["patient_uuid"] == active_id].iloc[0]

        st.json(patient.to_dict())

        current_loc = patient.get("current_location", "INTAKE")

        new_location = st.selectbox(
            "Move patient",
            COMPARTMENTS,
            index=COMPARTMENTS.index(current_loc)
            if current_loc in COMPARTMENTS else 0,
            key=f"move_{active_id}"
        )

        if st.button("Update Location", key=f"move_btn_{active_id}"):

            sheets = service.move_patient(sheets, active_id, new_location)

            sheets = scan_service.log_scan(
                sheets,
                active_id,
                action="MOVE_COMPARTMENT",
                source="streamlit_ui",
                metadata=new_location
            )

            backend.write_patients(sheets)
            refresh_state()
            st.rerun()


# directory tab
with tab_directory:

    st.subheader("Patient Search")

    df = st.session_state.df

    query = st.text_input("Search Patients", key="dir_search")

    if query:
        results = fuzzy_search_patients(df, query)
        df, sheets = log_event(
                patient_id=results.iloc[0]["patient_uuid"] if not results.empty else "N/A",
                action="DIRECTORY_SEARCH",
                metadata=query
            )
    else:
        results = df

    if results.empty:
        st.warning("No matching patients found")
    else:
        for _, p in results.iterrows():

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"""
                **{p['first_name']} {p['last_name']}**

                - ID: `{p['patient_uuid']}`
                - Location: `{p.get('current_location', 'INTAKE')}`
                """)

            with col2:
                qr = generate_qr(p["patient_uuid"])
                st.image(qr, width=100)


# log tab
with tab_logs:

    st.subheader("Event Logs")

    sheets = st.session_state.sheets

    if "ScanLog" in sheets:

        logs = sheets["ScanLog"].sort_values("timestamp", ascending=False)

        st.dataframe(logs)

        st.subheader("Movement Events")

        st.dataframe(
            logs[logs["action"] == "MOVE_COMPARTMENT"]
        )

        st.subheader("Lookup Events")

        st.dataframe(
            logs[logs["action"].isin(["SCAN_LOOKUP", "PATIENT_LOOKUP"])]
        )