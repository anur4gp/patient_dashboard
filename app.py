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

COMPARTMENT_COLORS = {
    "Intake":      "#4A90D9",
    "Waiting":     "#E6A817",
    "Doctor":      "#5BAD6F",
    "Pharmacy":    "#9B6FD4",
    "Discharged":  "#888888",
}

# ---------------------------------------------------------------------------
# Init services
# ---------------------------------------------------------------------------
backend = ExcelBackend(DATA_PATH)
service  = PatientService(backend)
event_log = EventLogService()

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _bootstrap():
    df, _ = service.get_patients()
    df = df.reset_index(drop=True)
    if "anon_token" not in df.columns:
        df["anon_token"] = [f"Patient #{i+1}" for i in df.index]
    if "compartment" not in df.columns:
        df["compartment"] = COMPARTMENTS[0]
    st.session_state.df = df

if "df" not in st.session_state:
    _bootstrap()

def get_df() -> pd.DataFrame:
    return st.session_state.df

# ---------------------------------------------------------------------------
# Core action
# ---------------------------------------------------------------------------
def move_patient(token: str, new_compartment: str):
    """
    Move patient to new_compartment. Returns error string or None on success.
    """
    df = get_df()
    idx = df.index[df["anon_token"] == token]
    if idx.empty:
        return "Patient not found."
    old = df.at[idx[0], "compartment"]
    if old == new_compartment:
        return f"Already in {new_compartment}."
    st.session_state.df.at[idx[0], "compartment"] = new_compartment
    event_log.log_move(from_compartment=old, to_compartment=new_compartment)
    return None

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


def compartment_badge(compartment: str) -> str:
    color = COMPARTMENT_COLORS.get(compartment, "#888")
    return (
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">'
        f'{compartment}</span>'
    )


def render_patient_card(p: pd.Series):
    skip = {"full_name", "anon_token", "compartment"}
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### {p['first_name']} {p['last_name']}")
        for key, val in p.to_dict().items():
            if key in ("first_name", "last_name") or key in skip:
                continue
            st.markdown(f"**{str(key).replace('_', ' ').title()}:** {val}")
    with col2:
        qr = generate_qr(p["patient_uuid"])
        st.image(qr, caption="Patient QR", width=120)


def render_move_control(token: str, current_compartment: str):
    """Compartment badge + inline move selector — used in Scan tab."""
    st.markdown(
        f"**Current location:** {compartment_badge(current_compartment)}",
        unsafe_allow_html=True,
    )
    st.write("")
    options = [c for c in COMPARTMENTS if c != current_compartment]
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        dest = st.selectbox(
            "Move to",
            options=options,
            key=f"move_dest_{token}",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Move →", key=f"move_btn_{token}", use_container_width=True):
            err = move_patient(token, dest)
            if err:
                st.warning(err)
            else:
                st.success(f"Moved to {dest}")
                st.rerun()

# ---------------------------------------------------------------------------
# Sidebar
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
tab_board, tab_scan, tab_directory = st.tabs(["🗂️ Board", "🔍 Scan", "👤 Directory"])

# ---------------------------------------------------------------------------
# Board — read-only flow summary
# ---------------------------------------------------------------------------
with tab_board:
    st.subheader("Patient Flow Summary")
    df = get_df()

    cols = st.columns(len(COMPARTMENTS))
    for col, compartment in zip(cols, COMPARTMENTS):
        count = int((df["compartment"] == compartment).sum())
        color = COMPARTMENT_COLORS[compartment]
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{color}22;
                    border:1.5px solid {color};
                    border-radius:10px;
                    padding:18px 10px;
                    text-align:center;
                ">
                    <div style="color:{color};font-weight:700;font-size:1em;">
                        {compartment}
                    </div>
                    <div style="font-size:2em;font-weight:800;margin-top:6px;">
                        {count}
                    </div>
                    <div style="color:#aaa;font-size:0.78em;">patient(s)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Add patient to Intake
    if st.button("➕ Add patient to Intake"):
        df = get_df()
        new_idx = len(df)
        new_token = f"Patient #{new_idx + 1}"
        new_row = pd.DataFrame([{
            "patient_uuid": f"anon-{new_idx}",
            "first_name":   "Unknown",
            "last_name":    "",
            "anon_token":   new_token,
            "compartment":  "Intake",
        }])
        st.session_state.df = pd.concat(
            [st.session_state.df, new_row], ignore_index=True
        )
        event_log.log_move(from_compartment="—", to_compartment="Intake")
        st.rerun()

# ---------------------------------------------------------------------------
# Scan — patient lookup + move control
# ---------------------------------------------------------------------------
with tab_scan:
    st.subheader("Scan Patient")
    scan_input = st.text_input(
        "Patient UUID",
        key="scan_input",
        placeholder="Scan or paste patient UUID...",
    )

    df = get_df()

    if scan_input:
        match = df[df["patient_uuid"] == scan_input]
        if match.empty:
            st.error("No patient found with that ID.")
        else:
            p = match.iloc[0]
            render_patient_card(p)
            st.divider()
            render_move_control(p["anon_token"], p["compartment"])

# ---------------------------------------------------------------------------
# Directory — fuzzy name search, read-only
# ---------------------------------------------------------------------------
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