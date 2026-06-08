import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime, timedelta
from services.excel_backend import ExcelBackend
from services.patient_service import PatientService
from services.qr_service import generate_qr
from services.scan_service import ScanService
from services.compartment_params import extract_params
from services.compartment_ode import solve, get_current_occupancy
from rapidfuzz import process, fuzz
from config import COMPARTMENTS, CLINIC_USERNAME, CLINIC_PASSWORD, CLINIC_NAME

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareTrack",
    page_icon="🏥",
    layout="wide",
)

# ── Compartment badge colours ──────────────────────────────────────────────
LOCATION_COLOURS = {
    "INTAKE":      "#6B7280",
    "WAITING":     "#D97706",
    "DOCTOR":      "#2563EB",
    "PHARMACY":    "#7C3AED",
    "DISCHARGED":  "#16A34A",
}

# ══════════════════════════════════════════════════════════════════════════
# LOGIN GATE
# Must run before any other content renders. st.stop() halts the rest
# of the script if the user is not authenticated.
# ══════════════════════════════════════════════════════════════════════════

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "login_staff" not in st.session_state:
    st.session_state.login_staff = ""

if not st.session_state.logged_in:
    # Centre the login card with empty columns either side
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"## 🏥 {CLINIC_NAME}")
        st.markdown("---")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign in", use_container_width=True):
            if username == CLINIC_USERNAME and password == CLINIC_PASSWORD:
                st.session_state.logged_in   = True
                st.session_state.login_staff = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    st.stop()   # Nothing below this renders until login succeeds


# ── Service init (only reached after login) ───────────────────────────────
backend      = ExcelBackend("data/mock_patients.xlsx")
service      = PatientService(backend)
scan_service = ScanService(backend)

# ── Session state bootstrap ───────────────────────────────────────────────
if "df" not in st.session_state or "sheets" not in st.session_state:
    df, sheets = service.get_patients()
    st.session_state.df     = df
    st.session_state.sheets = sheets

if "last_logged_query" not in st.session_state:
    st.session_state.last_logged_query = ""

# dir_selected_patient drives the inline record in the Directory tab.
# "Open record" sets this; closing the record clears it.
if "dir_selected_patient" not in st.session_state:
    st.session_state.dir_selected_patient = None


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def refresh_state():
    df, sheets = service.get_patients()
    st.session_state.df     = df
    st.session_state.sheets = sheets


def log_event(patient_id, action, metadata=None):
    sheets = st.session_state.sheets
    sheets = scan_service.log_scan(
        sheets, patient_id,
        action=action,
        source=f"ui:{st.session_state.login_staff}",
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
    colour = LOCATION_COLOURS.get(str(location).upper(), "#6B7280")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">'
        f'{location}</span>'
    )


def qr_to_base64(patient_uuid: str) -> str:
    """Convert QR image to base64 string for embedding in HTML.
    Handles whatever generate_qr returns: PIL Image, BytesIO, or a filepath string."""
    img = generate_qr(patient_uuid)

    # Case 1: PIL Image
    if hasattr(img, "save"):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # Case 2: BytesIO or file-like object
    if hasattr(img, "read"):
        img.seek(0)
        return base64.b64encode(img.read()).decode()

    # Case 3: string — either a filepath or an existing base64/data URL
    if isinstance(img, str):
        # Already a data URL like "data:image/png;base64,..."
        if img.startswith("data:"):
            return img.split(",", 1)[-1]
        # Assume it's a filepath
        with open(img, "rb") as f:
            return base64.b64encode(f.read()).decode()

    raise TypeError(f"generate_qr returned unexpected type: {type(img)}")


def render_patient_card(patient: pd.Series, show_qr: bool = False):
    """Structured card replacing the old st.json() dump."""
    loc = str(patient.get("current_location", "INTAKE"))
    st.markdown(
        f"### {patient.get('first_name', '')} {patient.get('last_name', '')} &nbsp;"
        + location_badge(loc),
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Patient ID",     patient.get("patient_uuid", "—"))
    col2.metric("Date of Birth",  patient.get("dob", "—"))
    col3.metric("Phone",          patient.get("phone", "—"))

    col4, col5, col6 = st.columns(3)
    col4.metric("Insurance",      patient.get("insurance_provider", "—"))
    col5.metric("Intake Date",    patient.get("intake_date", "—"))
    col6.metric("Assigned Staff", patient.get("assigned_staff", "—"))

    if show_qr:
        qr = generate_qr(patient["patient_uuid"])
        st.image(qr, width=140)

    with st.expander("All fields", expanded=False):
        skip = {"patient_uuid", "full_name"}
        fields = {k: v for k, v in patient.items() if k not in skip}
        keys = list(fields.keys())
        mid  = (len(keys) + 1) // 2
        c1, c2 = st.columns(2)
        for k in keys[:mid]:
            c1.markdown(f"**{k.replace('_',' ').title()}:** {fields[k]}")
        for k in keys[mid:]:
            c2.markdown(f"**{k.replace('_',' ').title()}:** {fields[k]}")


def render_dispense_panel(patient: pd.Series, tab_key: str):
    """
    Full pharmacy dispense mechanic.
    - Checks insurance status and blocks dispense if on hold.
    - Checks whether the patient has already been dispensed today.
    - On confirm, logs DISPENSE_CONFIRMED with staff name + timestamp.
    - Shows a summary of what was dispensed after confirmation.
    """
    st.markdown("#### 💊 Pharmacy verification")

    ins_status = str(patient.get("insurance_status", "")).upper()
    is_hold    = ins_status in ("ISSUE", "HOLD", "EXPIRED")

    pid      = patient.get("patient_uuid", "")
    med      = patient.get("current_medication", "—")
    dob      = patient.get("dob", "—")
    name     = f"{patient.get('first_name','')} {patient.get('last_name','')}"

    # ── Identity confirmation strip ───────────────────────────────────────
    vc1, vc2, vc3 = st.columns(3)
    vc1.markdown(f"**Patient:** {name}")
    vc2.markdown(f"**DOB:** {dob}")
    vc3.markdown(f"**Today's medication:** `{med}`")

    st.markdown("")  # spacer

    # ── Insurance status ──────────────────────────────────────────────────
    if is_hold:
        st.error(
            f"⚠️ **Insurance hold — do not dispense.** "
            f"Status: `{patient.get('insurance_status', 'UNKNOWN')}` · "
            f"Resolve with billing before proceeding."
        )
        return   # hard stop — no dispense button shown

    st.success("✅ Insurance verified.")

    # ── Check for same-day dispense already logged ────────────────────────
    sheets = st.session_state.sheets
    already_dispensed_today = False
    if "ScanLog" in sheets and not sheets["ScanLog"].empty:
        slog = sheets["ScanLog"].copy()
        slog["timestamp"] = pd.to_datetime(slog["timestamp"], errors="coerce")
        today_start = pd.Timestamp.now().normalize()
        prior = slog[
            (slog["patient_uuid"] == pid)
            & (slog["action"] == "DISPENSE_CONFIRMED")
            & (slog["timestamp"] >= today_start)
        ]
        if not prior.empty:
            already_dispensed_today = True
            last_time = prior["timestamp"].max().strftime("%-I:%M %p")
            last_by   = prior.iloc[-1].get("source", "unknown")
            st.warning(
                f"⚠️ **Already dispensed today at {last_time}** "
                f"(logged by `{last_by}`). "
                f"Confirm again only if re-dispensing is authorised."
            )

    # ── Dispense confirmation ─────────────────────────────────────────────
    btn_label = (
        "✅ Confirm & mark dispensed"
        if not already_dispensed_today
        else "⚠️ Dispense again (override)"
    )

    # Require a notes field on override to create an audit trail
    notes = ""
    if already_dispensed_today:
        notes = st.text_input(
            "Override reason (required)",
            key=f"dispense_notes_{pid}_{tab_key}",
            placeholder="e.g. first bag lost, replacement authorised by Dr. Chen",
        )

    btn_disabled = already_dispensed_today and notes.strip() == ""

    if st.button(btn_label, key=f"dispense_btn_{pid}_{tab_key}", disabled=btn_disabled):
        meta = f"med={med}"
        if notes:
            meta += f" | override_reason={notes.strip()}"
        log_event(pid, action="DISPENSE_CONFIRMED", metadata=meta)
        st.success(
            f"Dispense confirmed for **{name}** — `{med}` — "
            f"logged at {datetime.now().strftime('%-I:%M %p')} "
            f"by `{st.session_state.login_staff}`."
        )


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — logged-in state + logout
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### 🏥 {CLINIC_NAME}")
    st.caption(f"Signed in as **{st.session_state.login_staff}**")
    st.markdown("---")
    if st.button("Sign out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# APP HEADER + TABS
# ══════════════════════════════════════════════════════════════════════════

st.title("🏥 CareTrack")
st.caption("Clinical patient management · local build")

tab_scan, tab_directory, tab_logs, tab_flow, tab_print = st.tabs([
    "🔍 Scan",
    "👤 Directory",
    "📜 Logs",
    "📊 Patient Flow",
    "🖨️ Print QR",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — SCAN
# ══════════════════════════════════════════════════════════════════════════

with tab_scan:
    st.subheader("Scan Patient")
    st.caption(
        "Type a patient UUID, or scan a QR label with a USB/BT scanner — "
        "it acts as a keyboard and types the ID into the field automatically."
    )

    scan_input = st.text_input("Scan or enter Patient ID", key="scan_input")
    df     = st.session_state.df
    sheets = st.session_state.sheets

    if scan_input:
        match = df[df["patient_uuid"] == scan_input]
        if len(match) > 0:
            if st.session_state.get("active_patient") != scan_input:
                st.session_state.active_patient = scan_input
                log_event(scan_input, action="SCAN_LOOKUP", metadata="scan_tab")
        else:
            st.error(f"No patient found for ID: `{scan_input}`")

    active_id = st.session_state.get("active_patient")

    if active_id:
        df   = st.session_state.df
        rows = df[df["patient_uuid"] == active_id]

        if rows.empty:
            st.warning("Patient record not found — try scanning again.")
            st.stop()

        patient = rows.iloc[0]
        render_patient_card(patient)
        st.divider()
        render_dispense_panel(patient, tab_key="scan")
        st.divider()

        # ── Compartment mover ─────────────────────────────────────────────
        st.markdown("#### 📍 Move patient")
        current_loc = str(patient.get("current_location", "INTAKE"))
        new_location = st.selectbox(
            "Move to compartment",
            COMPARTMENTS,
            index=COMPARTMENTS.index(current_loc) if current_loc in COMPARTMENTS else 0,
            key=f"move_{active_id}",
        )
        if st.button("Update location", key=f"move_btn_{active_id}"):
            sheets = service.move_patient(sheets, active_id, new_location)
            sheets = scan_service.log_scan(
                sheets, active_id,
                action="MOVE_COMPARTMENT",
                source=f"ui:{st.session_state.login_staff}",
                metadata=new_location,
            )
            backend.write_patients(sheets)
            refresh_state()
            st.success(f"Moved to **{new_location}**.")
            st.rerun()

        with st.expander("Show QR code", expanded=False):
            qr = generate_qr(active_id)
            st.image(qr, width=180)
            st.caption(f"Patient ID: `{active_id}`")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DIRECTORY
# Streamlit does not support programmatic tab switching via button click,
# so "Open record" renders the full patient card inline within this tab
# rather than jumping to Scan. State is held in dir_selected_patient.
# ══════════════════════════════════════════════════════════════════════════

with tab_directory:
    st.subheader("Patient Directory")
    df    = st.session_state.df
    query = st.text_input("Search by name", key="dir_search")

    if query:
        results = fuzzy_search_patients(df, query)
        if query != st.session_state.last_logged_query and not results.empty:
            st.session_state.last_logged_query = query
            log_event(
                patient_id=results.iloc[0]["patient_uuid"],
                action="DIRECTORY_SEARCH",
                metadata=query,
            )
    else:
        results = df

    # ── Inline record panel ───────────────────────────────────────────────
    # Rendered above the list so it's immediately visible after clicking.
    selected_pid = st.session_state.dir_selected_patient
    if selected_pid:
        sel_rows = df[df["patient_uuid"] == selected_pid]
        if not sel_rows.empty:
            sel_patient = sel_rows.iloc[0]
            with st.container():
                close_col, _ = st.columns([1, 6])
                with close_col:
                    if st.button("✕ Close record", key="close_dir_record"):
                        st.session_state.dir_selected_patient = None
                        st.rerun()
                st.markdown("---")
                render_patient_card(sel_patient, show_qr=True)
                st.divider()
                render_dispense_panel(sel_patient, tab_key="dir")
                st.divider()

                # Compartment mover inline in directory
                st.markdown("#### 📍 Move patient")
                cur_loc = str(sel_patient.get("current_location", "INTAKE"))
                new_loc_dir = st.selectbox(
                    "Move to compartment",
                    COMPARTMENTS,
                    index=COMPARTMENTS.index(cur_loc) if cur_loc in COMPARTMENTS else 0,
                    key=f"dir_move_{selected_pid}",
                )
                if st.button("Update location", key=f"dir_move_btn_{selected_pid}"):
                    sheets = st.session_state.sheets
                    sheets = service.move_patient(sheets, selected_pid, new_loc_dir)
                    sheets = scan_service.log_scan(
                        sheets, selected_pid,
                        action="MOVE_COMPARTMENT",
                        source=f"ui:{st.session_state.login_staff}",
                        metadata=new_loc_dir,
                    )
                    backend.write_patients(sheets)
                    refresh_state()
                    st.success(f"Moved to **{new_loc_dir}**.")
                    st.rerun()
                st.markdown("---")

    # ── Patient list ──────────────────────────────────────────────────────
    if results.empty:
        st.warning("No matching patients found.")
    else:
        st.caption(f"{len(results)} patient(s) shown")
        for _, p in results.iterrows():
            col1, col2, col3 = st.columns([4, 1, 1])
            loc = str(p.get("current_location", "INTAKE"))
            with col1:
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
                if st.button("Open record", key=f"open_{p['patient_uuid']}"):
                    st.session_state.dir_selected_patient = p["patient_uuid"]
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

        today = pd.Timestamp.now().normalize()
        today_logs = logs[logs["timestamp"] >= today]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total events",  len(logs))
        m2.metric("Events today",  len(today_logs))
        m3.metric("Scans today",   len(today_logs[today_logs["action"] == "SCAN_LOOKUP"]))
        m4.metric("Dispenses today", len(today_logs[today_logs["action"] == "DISPENSE_CONFIRMED"]))

        st.divider()

        ACTION_OPTIONS = ["All"] + sorted(logs["action"].dropna().unique().tolist())
        filter_action  = st.selectbox("Filter by action", ACTION_OPTIONS, key="log_filter")
        date_range     = st.date_input(
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
                "timestamp":    st.column_config.DatetimeColumn("Time", format="MMM D, YYYY · h:mm a"),
                "action":       st.column_config.TextColumn("Action"),
                "patient_uuid": st.column_config.TextColumn("Patient ID"),
                "metadata":     st.column_config.TextColumn("Detail"),
                "source":       st.column_config.TextColumn("Staff / Source"),
            },
        )

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export filtered log as CSV",
            data=csv,
            file_name="caretrack_log_export.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PATIENT FLOW
# ══════════════════════════════════════════════════════════════════════════

with tab_flow:
    st.subheader("Patient Flow Model")
    st.caption(
        "Daily forecast of patient volume through each area, derived from "
        "historical movement data. Updates automatically as logs accumulate."
    )
    sheets = st.session_state.sheets

    if "ScanLog" not in sheets or sheets["ScanLog"].empty:
        st.info("The flow model needs movement data. Start scanning and moving patients.")
    else:
        log_df = sheets["ScanLog"].copy()

        with st.expander("⚙️ Model settings", expanded=False):
            lwbs_on = st.toggle("Account for patients who leave before being seen (LWBS)", value=True)
            if st.button("🔄 Recalculate now"):
                st.cache_data.clear()

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

        total_expected = int(np.trapz(sol.lambda_t, sol.t))

        def fmt_duration(mins):
            if mins <= 0: return "—"
            if mins < 60: return f"{mins:.0f} min"
            return f"{int(mins//60)}h {int(mins%60)}m"

        def fmt_hour(h):
            t = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            return (t + timedelta(hours=float(h))).strftime("%-I:%M %p")

        c1, c2, c3 = st.columns(3)
        c1.metric("Patients expected today",  total_expected)
        c2.metric("Avg time through clinic",  fmt_duration(sol.mean_system_time * 60))
        c3.metric("Predicted bottleneck", sol.bottleneck,
                  delta=f"peaks ~{fmt_hour(sol.peak_waiting_hour)}", delta_color="off")

        st.divider()

        import plotly.graph_objects as go
        base  = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        times = [base + timedelta(hours=float(h)) for h in sol.t]
        fig   = go.Figure()
        for name, y, lc, fc in [
            ("Waiting room", sol.W, "#D97706", "rgba(217,119,6,0.2)"),
            ("With doctor",  sol.D, "#2563EB", "rgba(37,99,235,0.2)"),
            ("Pharmacy",     sol.P, "#7C3AED", "rgba(124,58,237,0.2)"),
        ]:
            fig.add_trace(go.Scatter(
                x=times, y=y, name=name, fill="tozeroy",
                line=dict(color=lc, width=2), fillcolor=fc,
                hovertemplate="%{y:.1f} patients<extra>" + name + "</extra>",
            ))
        peak_time = base + timedelta(hours=float(sol.peak_waiting_hour))
        fig.add_vline(x=peak_time.isoformat(), line_dash="dot",
                      line_color="rgba(217,119,6,0.5)",
                      annotation_text="Peak waiting", annotation_position="top right",
                      annotation_font_size=11)
        fig.update_layout(
            xaxis_title="Time of day", yaxis_title="Patients in area",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(180,180,180,0.15)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Area status")

        def area_card(name, occ, rate, peak_hour=None):
            peak     = float(np.max(occ))
            mean_occ = float(np.mean(occ))
            ratio    = peak / mean_occ if mean_occ > 0 else 1
            dwell    = fmt_duration((1.0 / rate * 60) if rate > 0 else 0)
            if ratio < 1.8:
                icon, msg = "🟢", "Flow looks steady throughout the day."
            elif ratio < 2.8:
                hs = f" around **{fmt_hour(peak_hour)}**" if peak_hour else ""
                icon, msg = "🟡", f"Moderate demand spike expected{hs}."
            else:
                hs = f" near **{fmt_hour(peak_hour)}**" if peak_hour else ""
                icon, msg = "🔴", f"High spike predicted{hs} — consider extra staffing."
            return icon, name, msg, dwell, peak

        cards = [
            area_card("Waiting room", sol.W, params.alpha,            sol.peak_waiting_hour),
            area_card("Doctor area",  sol.D, params.beta+params.gamma, None),
            area_card("Pharmacy",     sol.P, params.delta,             None),
        ]
        for col, (icon, name, msg, dwell, peak) in zip(st.columns(3), cards):
            with col:
                st.markdown(f"**{icon} {name}**")
                st.markdown(msg)
                st.caption(f"Avg dwell: **{dwell}** · Predicted peak: **{peak:.0f}** patients")

        with st.expander("📋 Model data quality", expanded=False):
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"**Arrivals tracked:** {params.n_arrivals}")
                st.markdown(f"**Waiting transitions:** {params.n_waiting_transitions}")
                st.markdown(f"**Doctor transitions:** {params.n_doctor_transitions}")
                st.markdown(f"**Pharmacy transitions:** {params.n_pharmacy_transitions}")
            with cb:
                st.markdown(f"**Avg wait time:** {fmt_duration(params.mean_wait_min)}")
                st.markdown(f"**Avg doctor visit:** {fmt_duration(params.mean_doctor_min)}")
                st.markdown(f"**Avg pharmacy dwell:** {fmt_duration(params.mean_pharmacy_min)}")
                st.markdown(f"**% routed to pharmacy:** {params.p_pharmacy*100:.0f}%")
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce")
            days = max((log_df["timestamp"].max() - log_df["timestamp"].min()).days + 1, 1)
            if days < 7:
                st.warning(f"Only {days} day(s) of data — accuracy improves with more usage.")
            else:
                st.success(f"Model trained on {days} days of movement data.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — PRINT QR
# Renders a clean printable HTML label for any patient. The print button
# calls window.print() which lets the browser/OS handle paper size,
# margins, and label sheets — no extra libraries needed.
# ══════════════════════════════════════════════════════════════════════════

with tab_print:
    st.subheader("Print QR Labels")
    st.caption(
        "Search for a patient, then hit Print to send the label to your printer. "
        "Works with standard label sheets (Avery 5160, etc.) — set your printer "
        "to 'actual size' and margins to minimum for best results."
    )

    df        = st.session_state.df
    print_query = st.text_input("Search patient to print", key="print_search")
    print_results = fuzzy_search_patients(df, print_query) if print_query else df

    if print_results.empty:
        st.warning("No matching patients.")
    else:
        # Patient selector dropdown from search results
        name_map = {
            f"{r['first_name']} {r['last_name']} — {r['patient_uuid']}": r["patient_uuid"]
            for _, r in print_results.iterrows()
        }
        chosen_label = st.selectbox("Select patient", list(name_map.keys()), key="print_select")
        chosen_pid   = name_map[chosen_label]
        chosen_row   = df[df["patient_uuid"] == chosen_pid].iloc[0]
        chosen_name  = f"{chosen_row.get('first_name','')} {chosen_row.get('last_name','')}"
        chosen_dob   = chosen_row.get("dob", "—")

        # Preview
        prev_col, _ = st.columns([1, 2])
        with prev_col:
            st.markdown("**Label preview**")
            qr_img = generate_qr(chosen_pid)
            st.image(qr_img, width=160)
            st.markdown(f"**{chosen_name}**")
            st.markdown(f"DOB: {chosen_dob}")
            st.code(chosen_pid, language=None)

        # Build printable HTML — embedded QR as base64 so it works offline
        qr_b64 = qr_to_base64(chosen_pid)
        print_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          @media print {{
            body {{ margin: 0; }}
            .no-print {{ display: none; }}
          }}
          body {{
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
            background: white;
            color: #111;
          }}
          .label {{
            border: 1.5px solid #ccc;
            border-radius: 10px;
            padding: 20px 28px;
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            width: 260px;
          }}
          .label img {{
            width: 160px;
            height: 160px;
          }}
          .name {{
            font-size: 17px;
            font-weight: 700;
            text-align: center;
            margin-top: 4px;
          }}
          .dob {{
            font-size: 13px;
            color: #555;
          }}
          .pid {{
            font-family: monospace;
            font-size: 12px;
            background: #f3f3f3;
            border-radius: 4px;
            padding: 3px 10px;
            letter-spacing: 0.04em;
          }}
          .clinic {{
            font-size: 11px;
            color: #888;
            margin-top: 2px;
          }}
          .print-btn {{
            margin-top: 20px;
            padding: 10px 32px;
            font-size: 15px;
            background: #2563EB;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
          }}
        </style>
        </head>
        <body>
          <div class="label">
            <img src="data:image/png;base64,{qr_b64}" alt="QR code for {chosen_name}" />
            <div class="name">{chosen_name}</div>
            <div class="dob">DOB: {chosen_dob}</div>
            <div class="pid">{chosen_pid}</div>
            <div class="clinic">CareTrack Patient Label</div>
          </div>
          <button class="print-btn no-print" onclick="window.print()">🖨️ Print label</button>
        </body>
        </html>
        """

        components.html(print_html, height=420, scrolling=False)
        st.caption(
            "💡 Tip: in your browser print dialog, set margins to 'None' or 'Minimum' "
            "and disable headers/footers for a clean label."
        )
