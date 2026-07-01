from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime
from rapidfuzz import process, fuzz

from services.excel_backend import ExcelBackend
from services.patient_service import PatientService
from services.qr_service import generate_qr
from services.event_log_service import EventLogService
from services.medication_service import (
    MedicationService,
    ROUTES,
    EFFECTIVENESS_OPTIONS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPARTMENTS = ["Intake", "Waiting", "Doctor", "Pharmacy", "Discharged"]
DATA_PATH = "data/mock_patients.xlsx"

COMPARTMENT_COLORS = {
    "Intake":     "#4A90D9",
    "Waiting":    "#E6A817",
    "Doctor":     "#5BAD6F",
    "Pharmacy":   "#9B6FD4",
    "Discharged": "#888888",
}

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
backend    = ExcelBackend(DATA_PATH)
service    = PatientService(backend)
event_log  = EventLogService()
med_service = MedicationService()

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
        f"{compartment}</span>"
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


def move_patient(token: str, new_compartment: str):
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


def render_move_control(token: str, current_compartment: str):
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
# MAR rendering helpers
# ---------------------------------------------------------------------------
def render_rx_chain(chain: list[dict]):
    """Render a prescription chain with effectiveness badges and pricing."""
    for i, rx in enumerate(chain):
        eff = rx["effectiveness"]
        eff_colors = {
            "Pending":     "#888",
            "Effective":   "#5BAD6F",
            "Partial":     "#E6A817",
            "Ineffective": "#D94A4A",
        }
        eff_color = eff_colors.get(eff, "#888")
        label = "Initial Rx" if i == 0 else f"Follow-up #{i}"

        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"**{label} — {rx['drug_name']}** &nbsp;"
                    f'<span style="background:{eff_color};color:#fff;'
                    f'padding:2px 8px;border-radius:10px;font-size:0.8em;">'
                    f"{eff}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Dose: {rx['dose']} · Route: {rx['route']} · "
                    f"Frequency: {rx['frequency']} · "
                    f"Prescribed: {rx['prescribed_at'][:16].replace('T', ' ')}"
                )
                if rx["notes"]:
                    st.markdown(f"*Notes: {rx['notes']}*")
            with c2:
                price = rx.get("price_usd", "0.0")
                st.metric("Price", f"${float(price):.2f}")

            # Effectiveness update controls (only if still Pending)
            if rx["effectiveness"] == "Pending":
                with st.expander("Update effectiveness"):
                    new_eff = st.selectbox(
                        "Outcome",
                        [e for e in EFFECTIVENESS_OPTIONS if e != "Pending"],
                        key=f"eff_{rx['rx_id']}",
                    )
                    outcome_notes = st.text_input(
                        "Outcome notes (optional)",
                        key=f"eff_notes_{rx['rx_id']}",
                    )
                    if st.button("Save outcome", key=f"eff_save_{rx['rx_id']}"):
                        med_service.update_effectiveness(
                            rx["rx_id"],
                            new_eff,
                            notes=outcome_notes or None,
                        )
                        st.success("Outcome recorded.")
                        st.rerun()


def render_new_rx_form(patient_uuid: str, prior_rx_id: str = ""):
    """Form to prescribe a new medication (or follow-up)."""
    label = "Prescribe follow-up medication" if prior_rx_id else "Prescribe medication"
    with st.expander(f"➕ {label}", expanded=not bool(prior_rx_id)):
        drug   = st.text_input("Drug name", key=f"drug_{prior_rx_id}_{patient_uuid}")
        dose   = st.text_input("Dose (e.g. 500mg, 10mg/5mL)", key=f"dose_{prior_rx_id}_{patient_uuid}")
        route  = st.selectbox("Route", ROUTES, key=f"route_{prior_rx_id}_{patient_uuid}")
        freq   = st.text_input("Frequency (e.g. TID, PRN, Once)", key=f"freq_{prior_rx_id}_{patient_uuid}")
        notes  = st.text_area("Clinical notes", key=f"notes_{prior_rx_id}_{patient_uuid}", height=80)
        price  = st.number_input("Price (USD)", min_value=0.0, step=0.01,
                                 key=f"price_{prior_rx_id}_{patient_uuid}")

        if st.button("Prescribe", key=f"rx_btn_{prior_rx_id}_{patient_uuid}"):
            if not drug or not dose:
                st.warning("Drug name and dose are required.")
            else:
                new_id = med_service.prescribe(
                    patient_uuid=patient_uuid,
                    drug_name=drug,
                    dose=dose,
                    route=route,
                    frequency=freq,
                    notes=notes,
                    price_usd=price,
                    prior_rx_id=prior_rx_id,
                )
                st.success(f"Prescribed — Rx ID: {new_id}")
                st.rerun()


def render_mar(patient_uuid: str):
    """Full MAR view for one patient: all prescription chains + new Rx form."""
    records = med_service.get_for_patient(patient_uuid)

    if not records:
        st.info("No prescriptions recorded for this patient.")
        render_new_rx_form(patient_uuid)
        return

    # Group into chains: collect root prescriptions (no prior_rx_id)
    roots = [r for r in records if not r["prior_rx_id"]]
    seen_ids = set()

    for root in roots:
        chain = med_service.get_chain(root["rx_id"])
        for r in chain:
            seen_ids.add(r["rx_id"])

        render_rx_chain(chain)

        # If the last Rx in the chain is Ineffective or Partial, offer follow-up
        last = chain[-1]
        if last["effectiveness"] in ("Ineffective", "Partial"):
            render_new_rx_form(patient_uuid, prior_rx_id=last["rx_id"])

    # Show form for a brand-new first Rx
    st.divider()
    render_new_rx_form(patient_uuid)

    # Total cost summary
    total = sum(float(r.get("price_usd", 0)) for r in records)
    st.markdown(
        f'<div style="text-align:right;color:#aaa;font-size:0.9em;">'
        f"Total medication cost this visit: <strong>${total:.2f}</strong></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Downloads")
    log_entries = event_log.read_log()
    st.caption(f"Compartment log: {len(log_entries)} transitions")
    st.download_button(
        label="⬇️ event_log.csv",
        data=event_log.get_csv_bytes(),
        file_name=f"event_log_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    st.download_button(
        label="⬇️ medication_log.csv",
        data=med_service.get_csv_bytes(),
        file_name=f"medication_log_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    st.divider()
    if st.button("🗑️ Clear event log", type="secondary"):
        open(event_log.path, "w").write("timestamp,from_compartment,to_compartment\n")
        st.success("Event log cleared.")
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("Patient Dashboard")
tab_board, tab_scan, tab_pharmacy, tab_directory = st.tabs(
    ["🗂️ Board", "🔍 Scan", "💊 Pharmacy", "👤 Directory"]
)

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
                    background:{color}22;border:1.5px solid {color};
                    border-radius:10px;padding:18px 10px;text-align:center;">
                    <div style="color:{color};font-weight:700;font-size:1em;">
                        {compartment}</div>
                    <div style="font-size:2em;font-weight:800;margin-top:6px;">
                        {count}</div>
                    <div style="color:#aaa;font-size:0.78em;">patient(s)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.divider()
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
# Scan — patient lookup + move
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
# Pharmacy — MAR per patient
# ---------------------------------------------------------------------------
with tab_pharmacy:
    st.subheader("Medication Administration Record")
    df = get_df()

    # Patient selector: search by anon token or name
    search = st.text_input(
        "Find patient (name or token)",
        key="pharm_search",
        placeholder="e.g. Patient #3 or John Smith...",
    )

    selected_uuid = None

    if search:
        # Try exact anon token match first
        token_match = df[df["anon_token"].str.lower() == search.lower()]
        if not token_match.empty:
            p = token_match.iloc[0]
            selected_uuid = p["patient_uuid"]
            st.markdown(
                f"**{p['anon_token']}** — "
                f"{compartment_badge(p['compartment'])}",
                unsafe_allow_html=True,
            )
        else:
            # Fall back to fuzzy name search
            results = fuzzy_search(df, search, limit=5)
            if results.empty:
                st.warning("No matching patient.")
            elif len(results) == 1:
                p = results.iloc[0]
                selected_uuid = p["patient_uuid"]
                st.markdown(
                    f"**{p['first_name']} {p['last_name']}** ({p['anon_token']}) — "
                    f"{compartment_badge(p['compartment'])}",
                    unsafe_allow_html=True,
                )
            else:
                options = {
                    f"{r['first_name']} {r['last_name']} ({r['anon_token']})": r["patient_uuid"]
                    for _, r in results.iterrows()
                }
                choice = st.selectbox("Multiple matches — select patient", list(options.keys()))
                selected_uuid = options[choice]

    if selected_uuid:
        st.divider()
        render_mar(selected_uuid)

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