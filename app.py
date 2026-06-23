import streamlit as st
import pandas as pd
from datetime import datetime
from rapidfuzz import process, fuzz

from services.excel_backend import ExcelBackend
from services.patient_service import PatientService
from services.qr_service import generate_qr
from services.event_log_service import EventLogService

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPARTMENTS = ["Intake", "Waiting", "Doctor", "Pharmacy", "Discharged"]
DATA_PATH = "data/mock_patients.xlsx"

# ---------------------------------------------------------------------------
# Init services
# ---------------------------------------------------------------------------
backend = ExcelBackend(DATA_PATH)
service = PatientService(backend)
event_log = EventLogService()  # writes to event_log.csv

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _bootstrap():
    df, sheets = service.get_patients()
    st.session_state.df = df
    # Assign anonymous tokens if not already present
    if "anon_token" not in df.columns:
        df = df.reset_index(drop=True)
        df["anon_token"] = [f"Patient #{i+1}" for i in df.index]
        st.session_state.df = df
    # Ensure compartment column exists
    if "compartment" not in st.session_state.df.columns:
        st.session_state.df["compartment"] = COMPARTMENTS[0]

if "df" not in st.session_state:
    _bootstrap()

def get_df() -> pd.DataFrame:
    return st.session_state.df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fuzzy_search(df: pd.DataFrame, query: str, limit: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["full_name"] = (
        df["first_name"].astype(str).str.lower().str.strip()
        + " "
        + df["last_name"].astype(str).str.lower().str.strip()
    )
    results = process.extract(
        query.lower().strip(),
        df["full_name"].tolist(),
        scorer=fuzz.WRatio,
        limit=limit,
    )
    matched = [df.iloc[idx] for _, score, idx in results if score > 50]
    return pd.DataFrame(matched)


def render_patient_card(p: pd.Series):
    """Full patient info card with QR — used in Scan and Directory tabs."""
    skip = {"full_name", "anon_token", "compartment"}
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### {p['first_name']} {p['last_name']}")
        for key, val in p.to_dict().items():
            if key in ("first_name", "last_name") or key in skip:
                continue
            str(key).replace('_', ' ').title()
    with col2:
        qr = generate_qr(p["patient_uuid"])
        st.image(qr, caption="Patient QR", width=120)


def move_patient(token: str, new_compartment: str):
    """Move patient between compartments and log the event."""
    df = get_df()
    idx = df.index[df["anon_token"] == token]
    if idx.empty:
        return
    old_compartment = df.at[idx[0], "compartment"]
    if old_compartment == new_compartment:
        return
    st.session_state.df.at[idx[0], "compartment"] = new_compartment
    event_log.log_move(
        from_compartment=old_compartment,
        to_compartment=new_compartment,
    )

# ---------------------------------------------------------------------------
# Sidebar — event log download
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Event Log")
    log_entries = event_log.read_log()
    st.caption(f"{len(log_entries)} transitions recorded")
    st.download_button(
        label="⬇️ Download event_log.csv",
        data=event_log.get_csv_bytes(),
        file_name=f"event_log_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    if st.button("🗑️ Clear log", type="secondary"):
        open(event_log.path, "w").write("timestamp,from_compartment,to_compartment\n")
        st.success("Log cleared.")
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("Patient Dashboard")
tab_kanban, tab_scan, tab_directory = st.tabs(["🗂️ Board", "🔍 Scan", "👤 Directory"])

# --- Kanban board (anonymous) ---
with tab_kanban:
    st.subheader("Patient Flow Board")

    df = get_df()
    cols = st.columns(len(COMPARTMENTS))

    for col, compartment in zip(cols, COMPARTMENTS):
        with col:
            st.markdown(f"**{compartment}**")
            patients_here = df[df["compartment"] == compartment]["anon_token"].tolist()
            st.caption(f"{len(patients_here)} patient(s)")

            for token in patients_here:
                with st.container(border=True):
                    st.markdown(f"🪪 {token}")
                    next_options = [c for c in COMPARTMENTS if c != compartment]
                    dest = st.selectbox(
                        "Move to",
                        options=next_options,
                        key=f"dest_{token}",
                        label_visibility="collapsed",
                    )
                    if st.button("Move →", key=f"move_{token}"):
                        move_patient(token, dest)
                        st.rerun()

    # Add a new anonymous patient to Intake
    st.divider()
    if st.button("➕ Add patient to Intake"):
        df = get_df()
        new_idx = len(df)
        new_token = f"Patient #{new_idx + 1}"
        new_row = pd.DataFrame([{
            "patient_uuid": f"anon-{new_idx}",
            "first_name": "Unknown",
            "last_name": "",
            "anon_token": new_token,
            "compartment": "Intake",
        }])
        st.session_state.df = pd.concat(
            [st.session_state.df, new_row], ignore_index=True
        )
        event_log.log_move(from_compartment="—", to_compartment="Intake")
        st.rerun()

# --- QR scan lookup ---
with tab_scan:
    st.subheader("Scan Patient QR")
    scan_input = st.text_input(
        "Patient UUID",
        key="scan_input",
        placeholder="Scan or paste patient UUID...",
    )
    df = get_df()
    if scan_input:
        match = df[df["patient_uuid"] == scan_input]
        if not match.empty:
            st.success("Patient found")
            render_patient_card(match.iloc[0])
        else:
            st.error("No patient found with that ID.")

# --- Directory ---
with tab_directory:
    st.subheader("Patient Directory")
    query = st.text_input(
        "Search by name",
        key="dir_search",
        placeholder="e.g. John Smith...",
    )
    df = get_df()
    if query:
        results = fuzzy_search(df, query)
        if results.empty:
            st.warning("No matching patients found.")
        else:
            for _, p in results.iterrows():
                with st.expander(
                    f"{p['first_name']} {p['last_name']} — {p['patient_uuid']}"
                ):
                    render_patient_card(p)
    else:
        st.info("Enter a name above to search.")