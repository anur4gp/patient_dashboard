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
import plotly.graph_objects as go

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
if "scan_timestamps" not in st.session_state:
    st.session_state.scan_timestamps = {}

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

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add any columns that downstream tabs expect but may not exist
    in older versions of the spreadsheet. Safe to call on every load.
    """
    defaults = {
        "insurance_status":   "Pending",
        "insurance_provider": "",
        "insurance_member_id": "",
        "insurance_group":    "",
        "insurance_plan":     "",
        "current_medication": "",
        "intake_date":        "",
        "assigned_staff":     "",
        "dob":                "",
        "phone":              "",
        "current_location":   "INTAKE",
        "intake_source":      "",
        "intake_notes":       "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

# ── Session state bootstrap ───────────────────────────────────────────────
if "df" not in st.session_state or "sheets" not in st.session_state:
    df, sheets = service.get_patients()
    df = _ensure_columns(df)           # ← add this line
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
    df = _ensure_columns(df)
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

tab_scan, tab_directory, tab_logs, tab_flow, \
tab_print, tab_new, tab_board, tab_insurance, tab_eod = st.tabs([
    "🔍 Scan", "👤 Directory", "📜 Logs", "📊 Patient Flow",
    "🖨️ Print QR", "🆕 Register Patient", "🏥 Patient Board",
    "🔒 Insurance", "📋 End of Day",
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
    DEBOUNCE_SECONDS = 10

    if scan_input:
        match = df[df["patient_uuid"] == scan_input]
        if len(match) > 0:
            now = datetime.now()
            last_scan_time = st.session_state.scan_timestamps.get(scan_input)
            seconds_since  = (
                (now - last_scan_time).total_seconds()
                if last_scan_time else DEBOUNCE_SECONDS + 1
            )
            is_new_scan = (
                st.session_state.get("active_patient") != scan_input
                or seconds_since > DEBOUNCE_SECONDS
            )
            if is_new_scan:
                st.session_state.active_patient             = scan_input
                st.session_state.scan_timestamps[scan_input] = now
                log_event(scan_input, action="SCAN_LOOKUP", metadata="scan_tab")
            elif seconds_since <= DEBOUNCE_SECONDS:
                st.caption(
                    f"⚡ Already scanned {int(seconds_since)}s ago — "
                    f"showing record without re-logging."
                )
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

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — REGISTER NEW PATIENT
# ══════════════════════════════════════════════════════════════════════════

with tab_new:
    st.subheader("Register New Patient")
    st.caption(
        "Fill in the fields below and click Register. A unique ID and QR label "
        "will be generated automatically. Fields marked * are required."
    )

    df = st.session_state.df

    # ── Live duplicate check (outside the form so it updates on each keystroke) ──
    st.markdown("#### Check for existing record")
    dup_first = st.text_input("First name *", key="new_first")
    dup_last  = st.text_input("Last name *",  key="new_last")
    dup_dob   = st.text_input(
        "Date of birth * (MM/DD/YYYY)", key="new_dob", placeholder="03/14/1987"
    )

    duplicate_found = False
    if dup_first and dup_last and dup_dob:
        existing = df[
            (df["first_name"].astype(str).str.lower().str.strip() == dup_first.lower().strip())
            & (df["last_name"].astype(str).str.lower().str.strip() == dup_last.lower().strip())
            & (df["dob"].astype(str).str.strip() == dup_dob.strip())
        ]
        if not existing.empty:
            duplicate_found = True
            dup_row = existing.iloc[0]
            st.warning(
                f"⚠️ **{dup_first.strip()} {dup_last.strip()}** (DOB {dup_dob}) already exists — "
                f"ID: `{dup_row['patient_uuid']}`, "
                f"currently in **{dup_row.get('current_location', '—')}**. "
                f"Open their record in the Directory tab instead of creating a duplicate."
            )
        else:
            st.success("No existing record found — safe to register.")

    st.divider()

    # ── Registration form ─────────────────────────────────────────────────
    # Every widget inside st.form MUST have an explicit key= that is unique
    # across the entire app — Streamlit auto-generates IDs from (label, placeholder)
    # so two fields with the same label (e.g. two "Phone" inputs) will collide
    # unless keyed explicitly.
    with st.form("new_patient_form", clear_on_submit=True):

        st.markdown("#### Demographics")
        fc1, fc2 = st.columns(2)
        first_name = fc1.text_input("First name *",  key="form_first", value=dup_first)
        last_name  = fc2.text_input("Last name *",   key="form_last",  value=dup_last)

        fd1, fd2, fd3 = st.columns(3)
        dob      = fd1.text_input("Date of birth * (MM/DD/YYYY)", key="form_dob",      value=dup_dob)
        gender   = fd2.selectbox("Gender", key="form_gender",
                                 options=["", "Female", "Male", "Non-binary",
                                          "Prefer not to say", "Other"])
        pronouns = fd3.text_input("Pronouns", key="form_pronouns",
                                  placeholder="she/her · he/him · they/them")

        fe1, fe2 = st.columns(2)
        ethnicity = fe1.selectbox("Race / Ethnicity", key="form_ethnicity", options=[
            "", "Hispanic or Latino", "Black or African American", "White", "Asian",
            "Middle Eastern", "Native Hawaiian or Pacific Islander",
            "American Indian or Alaska Native", "Two or more races",
            "Prefer not to say", "Other",
        ])
        # Key distinguishes this from the EC phone field below
        pt_phone = fe2.text_input("Patient phone", key="form_pt_phone",
                                  placeholder="(916) 555-0100")

        ff1, ff2 = st.columns(2)
        alt_phone = ff1.text_input("Alt phone", key="form_alt_phone", placeholder="optional")
        email     = ff2.text_input("Email",     key="form_email",     placeholder="optional")

        fg1, fg2, fg3, fg4 = st.columns([3, 2, 1, 1])
        address  = fg1.text_input("Address",  key="form_address",  placeholder="412 Oak St")
        city     = fg2.text_input("City",     key="form_city",     placeholder="Sacramento")
        state    = fg3.text_input("State",    key="form_state",    placeholder="CA", max_chars=2)
        zip_code = fg4.text_input("ZIP",      key="form_zip",      placeholder="95814", max_chars=5)

        st.markdown("#### Emergency contact")
        fh1, fh2, fh3 = st.columns(3)
        ec_name  = fh1.text_input("EC name",         key="form_ec_name",  placeholder="Full name")
        ec_rel   = fh2.text_input("EC relationship", key="form_ec_rel",   placeholder="Spouse, Parent…")
        ec_phone = fh3.text_input("EC phone",        key="form_ec_phone", placeholder="(916) 555-0100")

        st.markdown("#### Insurance")
        fi1, fi2, fi3 = st.columns(3)
        ins_provider = fi1.text_input("Insurance provider", key="form_ins_provider",
                                      placeholder="Blue Cross, Medicaid…")
        ins_member   = fi2.text_input("Member ID",          key="form_ins_member",
                                      placeholder="BCX-0000000")
        ins_group    = fi3.text_input("Group number",       key="form_ins_group",
                                      placeholder="GRP-00000")

        fj1, fj2 = st.columns(2)
        ins_plan   = fj1.selectbox("Plan type", key="form_ins_plan",
                                   options=["", "PPO", "HMO", "EPO", "POS",
                                            "Medicaid", "Medicare", "Other"])
        ins_status = fj2.selectbox("Insurance status", key="form_ins_status",
                                   options=["Verified", "Pending", "ISSUE", "HOLD"])

        st.markdown("#### Intake")
        fk1, fk2, fk3 = st.columns(3)
        intake_source = fk1.selectbox("How did they come in? *", key="form_intake_source",
                                      options=["", "Referral", "Walk-in", "Tabling",
                                               "Online / Website", "Other"])
        intake_date    = fk2.text_input("Intake date *", key="form_intake_date",
                                        value=datetime.now().strftime("%m/%d/%Y"))
        assigned_staff = fk3.text_input("Assigned staff", key="form_assigned_staff",
                                        value=st.session_state.login_staff)

        current_med = st.text_input(
            "Current / initial medication", key="form_current_med",
            placeholder="e.g. Metformin 500mg — leave blank if not yet prescribed",
        )
        intake_notes = st.text_area(
            "Intake notes", key="form_intake_notes", height=100,
            placeholder="Any initial observations, referral source details, flags…",
        )

        st.markdown("")
        submitted = st.form_submit_button(
            "✅ Register patient",
            use_container_width=True,
            disabled=duplicate_found,
        )

    # ── On submit (outside the form block) ────────────────────────────────
    if submitted:
        errors = []
        if not first_name.strip():  errors.append("First name is required.")
        if not last_name.strip():   errors.append("Last name is required.")
        if not dob.strip():         errors.append("Date of birth is required.")
        if not intake_source:       errors.append("Intake source is required.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            import uuid as _uuid
            new_uuid = "PT-" + str(_uuid.uuid4())[:8].upper()

            new_row = {
                "patient_uuid":        new_uuid,
                "first_name":          first_name.strip(),
                "last_name":           last_name.strip(),
                "dob":                 dob.strip(),
                "gender":              gender,
                "pronouns":            pronouns.strip(),
                "race_ethnicity":      ethnicity,
                "phone":               pt_phone.strip(),
                "alt_phone":           alt_phone.strip(),
                "email":               email.strip(),
                "address":             address.strip(),
                "city":                city.strip(),
                "state":               state.strip().upper(),
                "zip":                 zip_code.strip(),
                "emergency_contact":   ec_name.strip(),
                "ec_relationship":     ec_rel.strip(),
                "ec_phone":            ec_phone.strip(),
                "insurance_provider":  ins_provider.strip(),
                "insurance_member_id": ins_member.strip(),
                "insurance_group":     ins_group.strip(),
                "insurance_plan":      ins_plan,
                "insurance_status":    ins_status,
                "intake_source":       intake_source,
                "intake_date":         intake_date.strip(),
                "assigned_staff":      assigned_staff.strip(),
                "current_medication":  current_med.strip(),
                "intake_notes":        intake_notes.strip(),
                "current_location":    "INTAKE",
            }

            sheets = st.session_state.sheets
            try:
                sheets = service.add_patient(sheets, new_row)
                backend.write_patients(sheets)
                log_event(new_uuid, action="PATIENT_CREATED",
                          metadata=f"source={intake_source}")
                refresh_state()

                st.success(
                    f"✅ **{first_name.strip()} {last_name.strip()}** registered. "
                    f"Patient ID: `{new_uuid}`"
                )
                qr_col, info_col = st.columns([1, 2])
                with qr_col:
                    qr = generate_qr(new_uuid)
                    st.image(qr, width=160)
                with info_col:
                    st.markdown(f"**Name:** {first_name.strip()} {last_name.strip()}")
                    st.markdown(f"**DOB:** {dob.strip()}")
                    st.markdown(f"**ID:** `{new_uuid}`")
                    st.markdown("**Location:** INTAKE")
                    st.caption(
                        "Head to the 🖨️ Print QR tab to print a label for this patient."
                    )
            except Exception as e:
                st.error(f"Failed to save patient: {e}")


# ── Shared helper: compute dwell time per patient from ScanLog ─────────────
# Used by both the board and the EOD report.
 
def get_dwell_times(sheets: dict) -> dict:
    """
    For each patient_uuid, return how long (in minutes) they have been
    in their current compartment, derived from the most recent
    MOVE_COMPARTMENT event in ScanLog.
    Returns {patient_uuid: minutes_in_current_location}.
    """
    if "ScanLog" not in sheets or sheets["ScanLog"].empty:
        return {}
    log = sheets["ScanLog"].copy()
    log["timestamp"] = pd.to_datetime(log["timestamp"], errors="coerce")
    moves = log[log["action"] == "MOVE_COMPARTMENT"].dropna(subset=["timestamp"])
    if moves.empty:
        return {}
    latest = (
        moves.sort_values("timestamp")
             .groupby("patient_uuid")
             .last()
             .reset_index()[["patient_uuid", "timestamp"]]
    )
    now = pd.Timestamp.now()
    latest["minutes"] = (now - latest["timestamp"]).dt.total_seconds() / 60
    return dict(zip(latest["patient_uuid"], latest["minutes"].round(0)))
 
 
def fmt_dwell(minutes: float) -> str:
    if minutes < 1:
        return "just arrived"
    if minutes < 60:
        return f"{int(minutes)}m"
    h, m = int(minutes // 60), int(minutes % 60)
    return f"{h}h {m}m" if m else f"{h}h"
 
 
def dwell_colour(minutes: float, compartment: str) -> str:
    """Traffic-light colouring based on expected dwell per compartment."""
    thresholds = {
        "INTAKE":     (15, 30),
        "WAITING":    (30, 60),
        "DOCTOR":     (20, 45),
        "PHARMACY":   (10, 25),
        "DISCHARGED": (0,  0),
    }
    warn, crit = thresholds.get(compartment.upper(), (30, 60))
    if minutes <= warn:  return "#16A34A"   # green
    if minutes <= crit:  return "#D97706"   # amber
    return "#DC2626"                        # red
 
 
# ══════════════════════════════════════════════════════════════════════════
# TAB — LIVE PATIENT BOARD
# ══════════════════════════════════════════════════════════════════════════
 
with tab_board:
    st.subheader("Live Patient Board")
    st.caption(
        "Real-time snapshot of every active patient by location. "
        "Dwell time colours: 🟢 on track · 🟡 watch · 🔴 overdue."
    )
 
    # Manual refresh + auto-refresh toggle
    bcol1, bcol2, bcol3 = st.columns([1, 1, 4])
    with bcol1:
        if st.button("🔄 Refresh", key="board_refresh"):
            refresh_state()
            st.rerun()
    with bcol2:
        show_discharged = st.toggle("Show discharged", value=False, key="board_discharged")
 
    df     = st.session_state.df
    sheets = st.session_state.sheets
    dwells = get_dwell_times(sheets)
 
    from config import COMPARTMENTS
    display_comps = COMPARTMENTS if show_discharged else [
        c for c in COMPARTMENTS if c != "DISCHARGED"
    ]
 
    # Build one column per compartment
    board_cols = st.columns(len(display_comps))
 
    for col, comp in zip(board_cols, display_comps):
        comp_patients = df[
            df["current_location"].astype(str).str.upper() == comp.upper()
        ]
        badge_colour = {
            "INTAKE":     "#6B7280",
            "WAITING":    "#D97706",
            "DOCTOR":     "#2563EB",
            "PHARMACY":   "#7C3AED",
            "DISCHARGED": "#16A34A",
        }.get(comp.upper(), "#6B7280")
 
        with col:
            # Compartment header with patient count pill
            st.markdown(
                f'<div style="background:{badge_colour};color:#fff;'
                f'padding:6px 12px;border-radius:8px;font-weight:600;'
                f'font-size:13px;margin-bottom:8px;text-align:center;">'
                f'{comp} &nbsp;'
                f'<span style="background:rgba(255,255,255,0.25);'
                f'padding:1px 8px;border-radius:12px;font-size:12px;">'
                f'{len(comp_patients)}</span></div>',
                unsafe_allow_html=True,
            )
 
            if comp_patients.empty:
                st.markdown(
                    '<p style="color:#9CA3AF;font-size:12px;text-align:center;'
                    'margin-top:12px;">No patients</p>',
                    unsafe_allow_html=True,
                )
            else:
                for _, p in comp_patients.iterrows():
                    pid      = p.get("patient_uuid", "")
                    name     = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                    dob      = p.get("dob", "—")
                    ins      = str(p.get("insurance_status", "")).upper()
                    ins_flag = " ⚠️" if ins in ("ISSUE", "HOLD", "EXPIRED") else ""
                    minutes  = dwells.get(pid, 0)
                    dc       = dwell_colour(minutes, comp)
                    dwell_str = fmt_dwell(minutes)
 
                    # Patient card
                    st.markdown(
                        f'<div style="border:1px solid #E5E7EB;border-radius:8px;'
                        f'padding:10px 12px;margin-bottom:8px;background:#FAFAFA;">'
                        f'<div style="font-weight:600;font-size:13px;">'
                        f'{name}{ins_flag}</div>'
                        f'<div style="font-size:11px;color:#6B7280;margin-top:2px;">'
                        f'DOB: {dob}</div>'
                        f'<div style="font-size:11px;color:#6B7280;">'
                        f'ID: {pid}</div>'
                        f'<div style="margin-top:6px;font-size:12px;font-weight:500;'
                        f'color:{dc};">⏱ {dwell_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
 
    st.divider()
 
    # ── Board summary metrics ─────────────────────────────────────────────
    active = df[df["current_location"].astype(str).str.upper() != "DISCHARGED"]
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Total active patients", len(active))
    sm2.metric("In waiting room", len(df[df["current_location"].astype(str).str.upper() == "WAITING"]))
    sm3.metric("With doctor",     len(df[df["current_location"].astype(str).str.upper() == "DOCTOR"]))
    sm4.metric("In pharmacy",     len(df[df["current_location"].astype(str).str.upper() == "PHARMACY"]))
 
    # Flag anyone overdue in waiting (> 60 min)
    overdue = [
        f"{row['first_name']} {row['last_name']} ({fmt_dwell(dwells.get(row['patient_uuid'], 0))})"
        for _, row in df[df["current_location"].astype(str).str.upper() == "WAITING"].iterrows()
        if dwells.get(row["patient_uuid"], 0) > 60
    ]
    if overdue:
        st.warning(
            f"⏰ **{len(overdue)} patient(s) waiting over 60 minutes:** "
            + " · ".join(overdue)
        )
 
 
# ══════════════════════════════════════════════════════════════════════════
# TAB — INSURANCE ISSUES
# ══════════════════════════════════════════════════════════════════════════
 
with tab_insurance:
    st.subheader("Insurance Issues")
    st.caption(
        "Patients with unresolved insurance flags. Sorted by urgency. "
        "Use the notes field to log follow-up actions."
    )
 
    df     = st.session_state.df
    sheets = st.session_state.sheets
 
    # Pull flagged patients
    flagged = df[
        df["insurance_status"].astype(str).str.upper().isin(
            ["ISSUE", "HOLD", "EXPIRED", "PENDING"]
        )
    ].copy()
 
    if flagged.empty:
        st.success("✅ No insurance issues on file.")
    else:
        # Compute days since intake
        flagged["intake_date_parsed"] = pd.to_datetime(
            flagged["intake_date"], errors="coerce"
        )
        flagged["days_pending"] = (
            pd.Timestamp.now() - flagged["intake_date_parsed"]
        ).dt.days.fillna(0).astype(int)
 
        # Sort: ISSUE/HOLD/EXPIRED first, then PENDING; within each group by days desc
        priority_order = {"ISSUE": 0, "HOLD": 0, "EXPIRED": 0, "PENDING": 1}
        flagged["_priority"] = (
            flagged["insurance_status"].astype(str).str.upper()
                   .map(priority_order).fillna(2)
        )
        flagged = flagged.sort_values(["_priority", "days_pending"], ascending=[True, False])
 
        # Summary strip
        ic1, ic2, ic3 = st.columns(3)
        hard_holds = flagged[flagged["insurance_status"].astype(str).str.upper()
                             .isin(["ISSUE", "HOLD", "EXPIRED"])]
        ic1.metric("Total flagged",  len(flagged))
        ic2.metric("Hard holds",     len(hard_holds),
                   delta="dispense blocked" if len(hard_holds) else None,
                   delta_color="inverse")
        ic3.metric("Pending review", len(flagged) - len(hard_holds))
 
        st.divider()
 
        # ── Per-patient rows ──────────────────────────────────────────────
        for _, p in flagged.iterrows():
            pid      = p.get("patient_uuid", "")
            name     = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            status   = str(p.get("insurance_status", "")).upper()
            provider = p.get("insurance_provider", "—")
            member   = p.get("insurance_member_id", "—")
            days     = int(p.get("days_pending", 0))
            loc      = str(p.get("current_location", "—"))
 
            is_hard  = status in ("ISSUE", "HOLD", "EXPIRED")
            row_bg   = "#000000" if is_hard else "#FFFBEB"
            status_colour = "#DC2626" if is_hard else "#D97706"
 
            with st.container():
                st.markdown(
                    f'<div style="background:{row_bg};border-radius:8px;'
                    f'padding:12px 16px;margin-bottom:4px;">'
                    f'<span style="font-weight:600;font-size:14px;">{name}</span>'
                    f'&nbsp;&nbsp;'
                    f'<span style="background:{status_colour};color:#fff;'
                    f'padding:2px 8px;border-radius:12px;font-size:11px;'
                    f'font-weight:600;">{status}</span>'
                    f'&nbsp;&nbsp;'
                    f'<span style="font-size:12px;color:#6B7280;">'
                    f'{provider} · Member: {member} · Currently: {loc} · '
                    f'{days}d pending</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
 
                note_col, btn_col = st.columns([4, 1])
                with note_col:
                    note = st.text_input(
                        "Follow-up note",
                        key=f"ins_note_{pid}",
                        placeholder="e.g. Called Aetna re: auth renewal — awaiting callback",
                        label_visibility="collapsed",
                    )
                with btn_col:
                    if st.button("Log note", key=f"ins_log_{pid}"):
                        if note.strip():
                            log_event(pid, action="INSURANCE_NOTE", metadata=note.strip())
                            st.success("Note logged.")
                        else:
                            st.warning("Enter a note first.")
 
                # Show prior notes from ScanLog
                if "ScanLog" in sheets and not sheets["ScanLog"].empty:
                    prior_notes = sheets["ScanLog"][
                        (sheets["ScanLog"]["patient_uuid"] == pid)
                        & (sheets["ScanLog"]["action"] == "INSURANCE_NOTE")
                    ].copy()
                    if not prior_notes.empty:
                        prior_notes["timestamp"] = pd.to_datetime(
                            prior_notes["timestamp"], errors="coerce"
                        )
                        prior_notes = prior_notes.sort_values("timestamp", ascending=False)
                        with st.expander(f"Prior notes ({len(prior_notes)})", expanded=False):
                            for _, n in prior_notes.iterrows():
                                ts  = n["timestamp"].strftime("%-m/%-d · %-I:%M %p") \
                                      if pd.notna(n["timestamp"]) else "—"
                                src = n.get("source", "—")
                                st.markdown(
                                    f'<div style="font-size:12px;padding:4px 0;'
                                    f'border-bottom:1px solid #E5E7EB;">'
                                    f'<span style="color:#6B7280;">{ts} · {src}</span><br>'
                                    f'{n.get("metadata","")}</div>',
                                    unsafe_allow_html=True,
                                )
 
                st.markdown("")  # spacer
 
 
# ══════════════════════════════════════════════════════════════════════════
# TAB — END OF DAY REPORT
# ══════════════════════════════════════════════════════════════════════════
 
with tab_eod:
    st.subheader("End of Day Report")
    st.caption("Summary of today's activity. Export as CSV or print for handoff.")
 
    sheets = st.session_state.sheets
    df     = st.session_state.df
 
    # Date picker — defaults to today, but allows reviewing past days
    report_date = st.date_input(
        "Report date", value=datetime.now().date(), key="eod_date"
    )
    day_start = pd.Timestamp(report_date)
    day_end   = day_start + timedelta(days=1)
 
    if "ScanLog" not in sheets or sheets["ScanLog"].empty:
        st.info("No log data available yet.")
    else:
        log = sheets["ScanLog"].copy()
        log["timestamp"] = pd.to_datetime(log["timestamp"], errors="coerce")
        today_log = log[
            (log["timestamp"] >= day_start) & (log["timestamp"] < day_end)
        ]
 
        # ── Headline metrics ──────────────────────────────────────────────
        patients_seen  = today_log["patient_uuid"].nunique()
        new_registered = len(today_log[today_log["action"] == "PATIENT_CREATED"])
        dispenses      = len(today_log[today_log["action"] == "DISPENSE_CONFIRMED"])
        ins_notes      = len(today_log[today_log["action"] == "INSURANCE_NOTE"])
        moves          = len(today_log[today_log["action"] == "MOVE_COMPARTMENT"])
 
        em1, em2, em3, em4, em5 = st.columns(5)
        em1.metric("Patients seen",       patients_seen)
        em2.metric("New registrations",   new_registered)
        em3.metric("Meds dispensed",      dispenses)
        em4.metric("Compartment moves",   moves)
        em5.metric("Insurance follow-ups", ins_notes)
 
        st.divider()
 
        # ── Avg dwell time per compartment ────────────────────────────────
        st.markdown("#### Average dwell times")
        st.caption("Time spent in each area, based on today's movement events.")
 
        move_log = today_log[today_log["action"] == "MOVE_COMPARTMENT"].copy()
        move_log["metadata"] = move_log["metadata"].astype(str).str.upper().str.strip()
        move_log = move_log.sort_values(["patient_uuid", "timestamp"])
        move_log["prev_ts"]   = move_log.groupby("patient_uuid")["timestamp"].shift(1)
        move_log["prev_comp"] = move_log.groupby("patient_uuid")["metadata"].shift(1)
        move_log["dwell_min"] = (
            move_log["timestamp"] - move_log["prev_ts"]
        ).dt.total_seconds() / 60
        move_log = move_log[
            (move_log["dwell_min"] > 0) & (move_log["dwell_min"] < 480)
        ]
 
        from config import COMPARTMENTS
        dwell_summary = []
        for comp in COMPARTMENTS:
            comp_rows = move_log[move_log["prev_comp"] == comp.upper()]
            if not comp_rows.empty:
                avg = comp_rows["dwell_min"].mean()
                mx  = comp_rows["dwell_min"].max()
                n   = len(comp_rows)
                dwell_summary.append({
                    "Compartment": comp,
                    "Avg dwell":   f"{avg:.0f} min",
                    "Max dwell":   f"{mx:.0f} min",
                    "Transitions": n,
                })
 
        if dwell_summary:
            st.dataframe(
                pd.DataFrame(dwell_summary),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No compartment movement data for this date.")
 
        st.divider()
 
        # ── Dispense log ──────────────────────────────────────────────────
        st.markdown("#### Medications dispensed")
        dispense_rows = today_log[today_log["action"] == "DISPENSE_CONFIRMED"].copy()
        if dispense_rows.empty:
            st.info("No dispenses recorded today.")
        else:
            dispense_rows["timestamp"] = dispense_rows["timestamp"].dt.strftime(
                "%-I:%M %p"
            )
            # Enrich with patient name
            dispense_rows = dispense_rows.merge(
                df[["patient_uuid", "first_name", "last_name"]],
                on="patient_uuid", how="left",
            )
            dispense_rows["Patient"] = (
                dispense_rows["first_name"].fillna("") + " "
                + dispense_rows["last_name"].fillna("")
            ).str.strip()
            st.dataframe(
                dispense_rows[["timestamp", "Patient", "patient_uuid", "metadata", "source"]]
                    .rename(columns={
                        "timestamp":    "Time",
                        "patient_uuid": "Patient ID",
                        "metadata":     "Medication / notes",
                        "source":       "Staff",
                    }),
                use_container_width=True,
                hide_index=True,
            )
 
        st.divider()
 
        # ── Still-active patients (not yet discharged) ────────────────────
        st.markdown("#### Patients still in system")
        still_in = df[
            df["current_location"].astype(str).str.upper() != "DISCHARGED"
        ][["first_name", "last_name", "patient_uuid", "current_location",
           "insurance_status"]]
 
        if still_in.empty:
            st.success("All patients discharged.")
        else:
            still_in["Name"] = (
                still_in["first_name"].fillna("") + " "
                + still_in["last_name"].fillna("")
            ).str.strip()
            st.dataframe(
                still_in[["Name", "patient_uuid", "current_location", "insurance_status"]]
                    .rename(columns={
                        "patient_uuid":      "ID",
                        "current_location":  "Location",
                        "insurance_status":  "Insurance",
                    }),
                use_container_width=True,
                hide_index=True,
            )
 
        st.divider()
 
        # ── Exports ───────────────────────────────────────────────────────
        st.markdown("#### Export")
        exp1, exp2 = st.columns(2)
 
        with exp1:
            csv_data = today_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download full day log (CSV)",
                data=csv_data,
                file_name=f"caretrack_eod_{report_date}.csv",
                mime="text/csv",
                key="eod_csv",
            )
 
        with exp2:
            # Printable HTML summary
            dwell_rows_html = "".join(
                f"<tr><td>{r['Compartment']}</td><td>{r['Avg dwell']}</td>"
                f"<td>{r['Max dwell']}</td><td>{r['Transitions']}</td></tr>"
                for r in dwell_summary
            ) if dwell_summary else "<tr><td colspan='4'>No data</td></tr>"
 
            still_in_html = "".join(
                f"<tr><td>{r['Name']}</td><td>{r['patient_uuid']}</td>"
                f"<td>{r['current_location']}</td></tr>"
                for _, r in still_in.iterrows()
            ) if not still_in.empty else "<tr><td colspan='3'>All discharged</td></tr>"
 
            print_report = f"""<!DOCTYPE html><html><head>
            <style>
              body{{font-family:Arial,sans-serif;padding:32px;color:#111;max-width:800px;margin:auto}}
              h1{{font-size:20px;margin-bottom:4px}} .sub{{color:#6B7280;font-size:13px;margin-bottom:24px}}
              h2{{font-size:15px;margin:24px 0 8px;border-bottom:1px solid #E5E7EB;padding-bottom:4px}}
              .metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:8px}}
              .metric{{border:1px solid #E5E7EB;border-radius:8px;padding:10px 12px}}
              .metric-val{{font-size:22px;font-weight:700}} .metric-lbl{{font-size:11px;color:#6B7280}}
              table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:4px}}
              th{{background:#F3F4F6;padding:6px 10px;text-align:left;font-size:11px}}
              td{{padding:6px 10px;border-bottom:1px solid #F3F4F6}}
              .print-btn{{margin-top:24px;padding:10px 28px;background:#2563EB;color:#fff;
                border:none;border-radius:6px;font-size:14px;cursor:pointer}}
              @media print{{.print-btn{{display:none}}}}
            </style></head><body>
            <h1>CareTrack · End of Day Report</h1>
            <div class="sub">{report_date.strftime('%B %-d, %Y')} · Generated {datetime.now().strftime('%-I:%M %p')}</div>
            <div class="metrics">
              <div class="metric"><div class="metric-val">{patients_seen}</div><div class="metric-lbl">Patients seen</div></div>
              <div class="metric"><div class="metric-val">{new_registered}</div><div class="metric-lbl">New registrations</div></div>
              <div class="metric"><div class="metric-val">{dispenses}</div><div class="metric-lbl">Meds dispensed</div></div>
              <div class="metric"><div class="metric-val">{moves}</div><div class="metric-lbl">Compartment moves</div></div>
              <div class="metric"><div class="metric-val">{ins_notes}</div><div class="metric-lbl">Insurance follow-ups</div></div>
            </div>
            <h2>Average dwell times</h2>
            <table><tr><th>Compartment</th><th>Avg</th><th>Max</th><th>Transitions</th></tr>
            {dwell_rows_html}</table>
            <h2>Patients still in system</h2>
            <table><tr><th>Name</th><th>ID</th><th>Location</th></tr>
            {still_in_html}</table>
            <button class="print-btn" onclick="window.print()">🖨️ Print report</button>
            </body></html>"""
 
            if st.button("🖨️ Open printable report", key="eod_print"):
                components.html(print_report, height=600, scrolling=True)