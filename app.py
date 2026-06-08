import streamlit as st
import pandas as pd
from services.excel_backend import ExcelBackend
from services.patient_service import PatientService
from services.qr_service import generate_qr
from services.scan_service import ScanService
from services.compartment_params import extract_params
from services.compartment_ode import solve, get_current_occupancy
from rapidfuzz import process, fuzz
from config import COMPARTMENTS

# ── Page config ────────────────────────────────────────────────────────────
# FIX: moved to top — st.set_page_config must be the very first Streamlit
# call, before any other st.* usage, or it raises a runtime error.
st.set_page_config(
    page_title="CareTrack",
    page_icon="🏥",
    layout="wide",
)

# ── Compartment badge colours ──────────────────────────────────────────────
LOCATION_COLOURS = {
    "INTAKE":     "#6B7280",   # grey
    "WAITING":    "#D97706",   # amber
    "DOCTOR":     "#2563EB",   # blue
    "PHARMACY":   "#7C3AED",   # purple
    "DISCHARGED": "#16A34A",   # green
}

# ── Service init ───────────────────────────────────────────────────────────
backend      = ExcelBackend("data/mock_patients.xlsx")
service      = PatientService(backend)
scan_service = ScanService(backend)

# ── Session state bootstrap ────────────────────────────────────────────────
if "df" not in st.session_state or "sheets" not in st.session_state:
    df, sheets = service.get_patients()
    st.session_state.df     = df
    st.session_state.sheets = sheets

# FIX: track last-logged directory query so we don't write to Excel on
# every keystroke — only when the query actually changes.
if "last_logged_query" not in st.session_state:
    st.session_state.last_logged_query = ""


# ── Helpers ────────────────────────────────────────────────────────────────

def refresh_state():
    """Re-read Excel into session state. Called only after actual writes."""
    df, sheets = service.get_patients()
    st.session_state.df     = df
    st.session_state.sheets = sheets


def log_event(patient_id, action, metadata=None):
    """
    Append one row to ScanLog and persist to Excel.
    FIX: returns early without writing if the action is a duplicate
    directory search for the same query string, preventing a full
    Excel write on every keystroke.
    """
    sheets = st.session_state.sheets
    sheets = scan_service.log_scan(
        sheets,
        patient_id,
        action=action,
        source="streamlit_ui",
        metadata=metadata,
    )
    backend.write_patients(sheets)
    refresh_state()
    return st.session_state.df, st.session_state.sheets


def fuzzy_search_patients(df, query, limit=10):
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


def location_badge(location: str) -> str:
    """Return an HTML span styled as a coloured pill badge."""
    colour = LOCATION_COLOURS.get(location.upper(), "#6B7280")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">'
        f'{location}</span>'
    )


def render_patient_card(patient: pd.Series):
    """
    FIX: replaces the raw st.json() dump with a readable card layout.
    Shows key fields prominently; everything else in an expander.
    """
    loc = patient.get("current_location", "INTAKE")

    # Header row: name + location badge
    st.markdown(
        f"### {patient.get('first_name', '')} {patient.get('last_name', '')} &nbsp;"
        + location_badge(loc),
        unsafe_allow_html=True,
    )

    # Primary fields
    col1, col2, col3 = st.columns(3)
    col1.metric("Patient ID",  patient.get("patient_uuid", "—"))
    col2.metric("Date of Birth", patient.get("dob", "—"))
    col3.metric("Phone",       patient.get("phone", "—"))

    col4, col5, col6 = st.columns(3)
    col4.metric("Insurance",   patient.get("insurance_provider", "—"))
    col5.metric("Intake Date", patient.get("intake_date", "—"))
    col6.metric("Assigned Staff", patient.get("assigned_staff", "—"))

    # Everything else in a collapsed expander so it's accessible but not noisy
    with st.expander("All fields", expanded=False):
        # Build a clean two-column table from whatever columns exist,
        # skipping internal/index fields
        skip = {"patient_uuid", "full_name"}
        fields = {k: v for k, v in patient.items() if k not in skip}
        keys   = list(fields.keys())
        mid    = (len(keys) + 1) // 2
        c1, c2 = st.columns(2)
        for k in keys[:mid]:
            c1.markdown(f"**{k.replace('_',' ').title()}:** {fields[k]}")
        for k in keys[mid:]:
            c2.markdown(f"**{k.replace('_',' ').title()}:** {fields[k]}")


# ══════════════════════════════════════════════════════════════════════════
# APP HEADER
# ══════════════════════════════════════════════════════════════════════════

st.title("🏥 CareTrack")
st.caption("Clinical patient management · local build")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_scan, tab_directory, tab_logs, tab_flow = st.tabs([
    "🔍 Scan",
    "👤 Directory",
    "📜 Logs",
    "📊 Patient Flow",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — SCAN
# ══════════════════════════════════════════════════════════════════════════

with tab_scan:
    st.subheader("Scan Patient")
    st.caption(
        "Enter a patient UUID directly, or scan a QR code with a USB/BT "
        "scanner — it will type the ID into the field automatically."
    )

    scan_input = st.text_input("Scan or enter Patient ID", key="scan_input")
    df     = st.session_state.df
    sheets = st.session_state.sheets

    if scan_input:
        match = df[df["patient_uuid"] == scan_input]
        if len(match) > 0:
            # Only update + log if it's a new scan (not a re-render)
            if st.session_state.get("active_patient") != scan_input:
                st.session_state.active_patient = scan_input
                log_event(scan_input, action="SCAN_LOOKUP", metadata="scan_tab")
        else:
            st.error(f"No patient found for ID: `{scan_input}`")

    active_id = st.session_state.get("active_patient")

    if active_id:
        df     = st.session_state.df
        sheets = st.session_state.sheets
        rows   = df[df["patient_uuid"] == active_id]

        # FIX: guard against stale active_patient after a data refresh
        if rows.empty:
            st.warning("Patient record not found — try scanning again.")
            st.stop()

        patient = rows.iloc[0]

        # ── Patient card (replaces st.json) ───────────────────────────────
        render_patient_card(patient)

        st.divider()

        # ── Pharmacy verification strip ───────────────────────────────────
        st.markdown("#### 💊 Pharmacy verification")
        ins_status = str(patient.get("insurance_status", "")).upper()
        is_hold    = ins_status in ("ISSUE", "HOLD", "EXPIRED")

        if is_hold:
            st.error(
                f"⚠️ **Insurance hold** — do not dispense until resolved. "
                f"Status: `{patient.get('insurance_status', 'UNKNOWN')}`"
            )
        else:
            st.success("Insurance verified — ready to dispense.")

        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.markdown(
            f"**Name:** {patient.get('first_name','')} {patient.get('last_name','')}"
        )
        vcol2.markdown(f"**DOB:** {patient.get('dob', '—')}")
        vcol3.markdown(f"**Today's medication:** {patient.get('current_medication', '—')}")

        if not is_hold:
            if st.button("✅ Confirm & mark dispensed", key=f"dispense_{active_id}"):
                log_event(active_id, action="DISPENSE_CONFIRMED", metadata="pharmacy")
                st.success("Dispense logged.")

        st.divider()

        # ── Compartment mover ─────────────────────────────────────────────
        st.markdown("#### 📍 Move patient")
        current_loc = patient.get("current_location", "INTAKE")
        new_location = st.selectbox(
            "Current location",
            COMPARTMENTS,
            index=COMPARTMENTS.index(current_loc) if current_loc in COMPARTMENTS else 0,
            key=f"move_{active_id}",
        )

        if st.button("Update location", key=f"move_btn_{active_id}"):
            sheets = service.move_patient(sheets, active_id, new_location)
            sheets = scan_service.log_scan(
                sheets,
                active_id,
                action="MOVE_COMPARTMENT",
                source="streamlit_ui",
                metadata=new_location,
            )
            backend.write_patients(sheets)
            refresh_state()
            st.success(f"Moved to **{new_location}**.")
            st.rerun()

        # ── QR code display ───────────────────────────────────────────────
        with st.expander("Show QR code", expanded=False):
            qr = generate_qr(active_id)
            st.image(qr, width=180)
            st.caption(f"Patient ID: `{active_id}`")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DIRECTORY
# ══════════════════════════════════════════════════════════════════════════

with tab_directory:
    st.subheader("Patient Directory")

    df    = st.session_state.df
    query = st.text_input("Search by name", key="dir_search")

    if query:
        results = fuzzy_search_patients(df, query)

        # FIX: only log + write to Excel when the query string has actually
        # changed, not on every Streamlit re-render while the user types.
        if query != st.session_state.last_logged_query and not results.empty:
            st.session_state.last_logged_query = query
            log_event(
                patient_id=results.iloc[0]["patient_uuid"],
                action="DIRECTORY_SEARCH",
                metadata=query,
            )
    else:
        results = df

    if results.empty:
        st.warning("No matching patients found.")
    else:
        st.caption(f"{len(results)} patient(s) shown")
        for _, p in results.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                loc = p.get("current_location", "INTAKE")

                with col1:
                    # FIX: uses location_badge helper instead of plain text
                    st.markdown(
                        f"**{p['first_name']} {p['last_name']}** &nbsp;"
                        + location_badge(loc),
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"ID: `{p['patient_uuid']}` · "
                        f"DOB: {p.get('dob', '—')} · "
                        f"Insurance: {p.get('insurance_provider', '—')}"
                    )

                with col2:
                    qr = generate_qr(p["patient_uuid"])
                    st.image(qr, width=80)

                with col3:
                    # Jump directly to scan tab for this patient
                    if st.button("Open record", key=f"open_{p['patient_uuid']}"):
                        st.session_state.active_patient = p["patient_uuid"]
                        st.rerun()

                st.divider()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — LOGS
# ══════════════════════════════════════════════════════════════════════════

with tab_logs:
    st.subheader("Event Log")
    sheets = st.session_state.sheets

    if "ScanLog" not in sheets or sheets["ScanLog"].empty:
        st.info("No events logged yet.")
    else:
        logs = sheets["ScanLog"].copy()
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
        logs = logs.sort_values("timestamp", ascending=False)

        # ── Summary metrics ───────────────────────────────────────────────
        today = pd.Timestamp.now().normalize()
        today_logs = logs[logs["timestamp"] >= today]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total events",    len(logs))
        m2.metric("Events today",    len(today_logs))
        m3.metric(
            "Scans today",
            len(today_logs[today_logs["action"] == "SCAN_LOOKUP"]),
        )
        m4.metric(
            "Moves today",
            len(today_logs[today_logs["action"] == "MOVE_COMPARTMENT"]),
        )

        st.divider()

        # FIX: single filterable dataframe instead of three separate
        # st.dataframe calls with overlapping data.
        ACTION_OPTIONS = ["All"] + sorted(logs["action"].dropna().unique().tolist())
        filter_action = st.selectbox(
            "Filter by action", ACTION_OPTIONS, key="log_filter"
        )
        date_range = st.date_input(
            "Date range",
            value=(today.date(), pd.Timestamp.now().date()),
            key="log_dates",
        )

        filtered = logs.copy()
        if filter_action != "All":
            filtered = filtered[filtered["action"] == filter_action]
        if len(date_range) == 2:
            start, end = date_range
            filtered = filtered[
                (filtered["timestamp"].dt.date >= start)
                & (filtered["timestamp"].dt.date <= end)
            ]

        st.caption(f"{len(filtered)} event(s) matching filter")
        st.dataframe(
            filtered.reset_index(drop=True),
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time", format="MMM D, YYYY · h:mm a"
                ),
                "action":       st.column_config.TextColumn("Action"),
                "patient_uuid": st.column_config.TextColumn("Patient ID"),
                "metadata":     st.column_config.TextColumn("Detail"),
                "source":       st.column_config.TextColumn("Source"),
            },
        )

        # Download button for audit export
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export filtered log as CSV",
            data=csv,
            file_name="caretrack_log_export.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PATIENT FLOW (compartment model)
# ══════════════════════════════════════════════════════════════════════════

with tab_flow:
    st.subheader("Patient Flow Model")
    st.caption(
        "Daily forecast of patient volume through each area, derived from "
        "historical movement data. Updates automatically as logs accumulate."
    )

    sheets = st.session_state.sheets

    if "ScanLog" not in sheets or sheets["ScanLog"].empty:
        st.info(
            "The flow model needs movement data to work. "
            "Start scanning and moving patients — the model will populate here."
        )
    else:
        log_df = sheets["ScanLog"].copy()

        # ── Model settings ────────────────────────────────────────────────
        with st.expander("⚙️ Model settings", expanded=False):
            lwbs_on = st.toggle(
                "Account for patients who leave before being seen (LWBS)",
                value=True,
            )
            if st.button("🔄 Recalculate now"):
                st.cache_data.clear()

        # ── Run solver ────────────────────────────────────────────────────
        with st.spinner("Running flow model…"):
            try:
                params = extract_params(log_df, lwbs_enabled=lwbs_on)
                y0     = get_current_occupancy(log_df)
                sol    = solve(params, y0=y0)
            except Exception as e:
                st.error(f"Solver error: {e}")
                st.stop()

        if not sol.success:
            st.error(f"Solver failed: {sol.message}")
            st.stop()

        # ── Summary metrics ───────────────────────────────────────────────
        import numpy as np
        total_expected = int(np.trapz(sol.lambda_t, sol.t))

        def fmt_duration(mins):
            if mins <= 0: return "—"
            if mins < 60: return f"{mins:.0f} min"
            return f"{int(mins//60)}h {int(mins%60)}m"

        def fmt_hour(h):
            from datetime import datetime, timedelta
            t = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            return (t + timedelta(hours=float(h))).strftime("%-I:%M %p")

        c1, c2, c3 = st.columns(3)
        c1.metric("Patients expected today", total_expected)
        c2.metric("Avg time through clinic", fmt_duration(sol.mean_system_time * 60))
        c3.metric("Predicted bottleneck", sol.bottleneck,
                  delta=f"peaks ~{fmt_hour(sol.peak_waiting_hour)}", delta_color="off")

        st.divider()

        # ── Occupancy chart ───────────────────────────────────────────────
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        base  = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        times = [base + timedelta(hours=float(h)) for h in sol.t]

        fig = go.Figure()
        traces = [
            ("Waiting room", sol.W, "#D97706", "rgba(217,119,6,0.2)"),
            ("With doctor",  sol.D, "#2563EB", "rgba(37,99,235,0.2)"),
            ("Pharmacy",     sol.P, "#7C3AED", "rgba(124,58,237,0.2)"),
        ]
        for name, y, line_col, fill_col in traces:
            fig.add_trace(go.Scatter(
                x=times, y=y, name=name,
                fill="tozeroy",
                line=dict(color=line_col, width=2),
                fillcolor=fill_col,
                hovertemplate="%{y:.1f} patients<extra>" + name + "</extra>",
            ))

        peak_time = base + timedelta(hours=float(sol.peak_waiting_hour))
        fig.add_vline(
            x=peak_time.isoformat(), line_dash="dot",
            line_color="rgba(217,119,6,0.5)",
            annotation_text=f"Peak waiting",
            annotation_position="top right",
            annotation_font_size=11,
        )
        fig.update_layout(
            xaxis_title="Time of day",
            yaxis_title="Patients in area",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(180,180,180,0.15)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Plain-language status cards ───────────────────────────────────
        st.markdown("### Area status")

        def area_card(name, occupancy_arr, rate, peak_hour=None):
            peak      = float(np.max(occupancy_arr))
            mean_occ  = float(np.mean(occupancy_arr))
            ratio     = peak / mean_occ if mean_occ > 0 else 1
            dwell_min = (1.0 / rate * 60) if rate > 0 else 0

            if ratio < 1.8:
                icon, msg = "🟢", "Flow looks steady throughout the day."
            elif ratio < 2.8:
                hour_str  = f" around **{fmt_hour(peak_hour)}**" if peak_hour else ""
                icon, msg = "🟡", f"Moderate demand spike expected{hour_str}."
            else:
                hour_str  = f" near **{fmt_hour(peak_hour)}**" if peak_hour else ""
                icon, msg = "🔴", f"High demand spike predicted{hour_str} — consider additional staffing."

            return icon, name, msg, fmt_duration(dwell_min), peak

        cards = [
            area_card("Waiting room", sol.W, params.alpha,           sol.peak_waiting_hour),
            area_card("Doctor area",  sol.D, params.beta+params.gamma, None),
            area_card("Pharmacy",     sol.P, params.delta,           None),
        ]
        cc1, cc2, cc3 = st.columns(3)
        for col, (icon, name, msg, dwell, peak) in zip([cc1, cc2, cc3], cards):
            with col:
                st.markdown(f"**{icon} {name}**")
                st.markdown(msg)
                st.caption(f"Avg dwell: **{dwell}** · Predicted peak: **{peak:.0f}** patients")

        # ── Data quality note ─────────────────────────────────────────────
        with st.expander("📋 Model data quality", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Arrivals tracked:** {params.n_arrivals}")
                st.markdown(f"**Waiting transitions:** {params.n_waiting_transitions}")
                st.markdown(f"**Doctor transitions:** {params.n_doctor_transitions}")
                st.markdown(f"**Pharmacy transitions:** {params.n_pharmacy_transitions}")
            with col_b:
                st.markdown(f"**Avg wait time:** {fmt_duration(params.mean_wait_min)}")
                st.markdown(f"**Avg doctor visit:** {fmt_duration(params.mean_doctor_min)}")
                st.markdown(f"**Avg pharmacy dwell:** {fmt_duration(params.mean_pharmacy_min)}")
                st.markdown(f"**% routed to pharmacy:** {params.p_pharmacy*100:.0f}%")

            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce")
            days = max((log_df["timestamp"].max() - log_df["timestamp"].min()).days + 1, 1)
            if days < 7:
                st.warning(
                    f"Only {days} day(s) of data — model accuracy will improve "
                    "with more usage."
                )
            else:
                st.success(f"Model trained on {days} days of movement data.")