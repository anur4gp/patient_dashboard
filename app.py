import streamlit as st
import pandas as pd
from services.excel_backend import ExcelBackend
from services.patient_service import PatientService
from services.qr_service import generate_qr
from services.scan_service import ScanService
from services.label_service import generate_printable_label
from services.auth_service import login
from rapidfuzz import process, fuzz
from config import COMPARTMENTS

from services.parameter_estimator import ParameterEstimator
from services.clinic_ode import ClinicODESystem
from services.ode_visualizer import ODEVisualizer
from services.arrival_model import ArrivalModel

from services.settings_service import SettingsService

# getting to login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# login gate
if not st.session_state.authenticated:

    st.title("🏥 Clinic Patient Dashboard")

    st.subheader("Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if login(username, password):
            st.session_state.authenticated = True
            st.rerun()

        else:
            st.error("Invalid username or password")

    st.stop()

# init
backend = ExcelBackend("data/mock_patients.xlsx")
service = PatientService(backend)
scan_service = ScanService(backend)
settings_service = SettingsService()

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


col1, col2 = st.columns([8, 1])

with col1:
    st.title("Patient Dashboard")

with col2:
    if st.button("Exit"):
        st.session_state.authenticated = False
        st.rerun()

# tabs
tab_scan, tab_directory, tab_forecast, tab_settings, tab_logs = st.tabs([
    "🔍 Scan",
    "👤 Directory",
    "📈 Forecast",
    "⚙️ Settings",
    "📜 Logs"
])

# tab 1: scan
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

        st.subheader("👤 Patient Profile")

        with st.container(border=True):

            col1, col2 = st.columns(2)

            with col1:
                first_name = st.text_input(
                    "First Name",
                    value=str(patient.get("first_name", "")),
                    key=f"fname_{active_id}"
                )

                last_name = st.text_input(
                    "Last Name",
                    value=str(patient.get("last_name", "")),
                    key=f"lname_{active_id}"
                )

                dob = st.text_input(
                    "Date of Birth",
                    value=str(patient.get("dob", "")),
                    key=f"dob_{active_id}"
                )

                sex = st.text_input(
                    "Sex",
                    value=str(patient.get("sex", "")),
                    key=f"sex_{active_id}"
                )

            with col2:

                phone = st.text_input(
                    "Phone",
                    value=str(patient.get("phone", "")),
                    key=f"phone_{active_id}"
                )

                provider = st.text_input(
                    "Provider",
                    value=str(patient.get("provider", "")),
                    key=f"provider_{active_id}"
                )

                insurance = st.text_input(
                    "Insurance",
                    value=str(patient.get("insurance", "")),
                    key=f"insurance_{active_id}"
                )

                allergies = st.text_area(
                    "Allergies",
                    value=str(patient.get("allergies", "")),
                    key=f"allergy_{active_id}"
                )

            medications = st.text_area(
                "Medications",
                value=str(patient.get("medications", "")),
                key=f"meds_{active_id}"
            )

            if st.button("💾 Save Patient Changes",
                        key=f"save_{active_id}"):

                updated_fields = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "dob": dob,
                    "sex": sex,
                    "phone": phone,
                    "provider": provider,
                    "insurance": insurance,
                    "allergies": allergies,
                    "medications": medications
                }

                sheets = service.update_patient(
                    sheets,
                    active_id,
                    updated_fields
                )

                backend.write_patients(sheets)
                refresh_state()

                st.success("Patient updated successfully")
                st.rerun()

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


# tab 2: directory
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

                if st.button(
                    "🖨 Print Label",
                    key=f"print_{p['patient_uuid']}"
                ):
                    generate_printable_label(p, qr)

                    st.success(
                        f"Opened printable label for "
                        f"{p['first_name']} {p['last_name']}"
                    )


# tab 3: operations
with tab_forecast:

    st.subheader("Clinic Operations Forecast")

    # refresh live data
    df, sheets = service.get_patients()
    st.session_state.df = df
    st.session_state.sheets = sheets

    # load settings
    settings = settings_service.load_settings()

    total_doctors = settings["total_doctors"]

    # estimate params
    estimator = ParameterEstimator(sheets)

    rates = estimator.estimate_transition_rates()

    pharmacy_probability = (
        estimator
        .estimate_pharmacy_probability()
    )

    # build arrival function
    try:

        arrival_model = ArrivalModel(sheets)

        lambda_function = (
            arrival_model
            .build_lambda_function()
        )

    except Exception:

        st.warning(
            "Not enough historical scan data "
            "to estimate arrivals yet."
        )

        lambda_function = (
            lambda t: 1
        )

    # build ode
    ode = ClinicODESystem(
        rates=rates,
        arrival_function=lambda_function,
        pharmacy_probability=pharmacy_probability,
        total_doctors=total_doctors
    )

    solution = ode.solve(
        initial_state=[0, 0, 0, 0, 0],
        t_end=600
    )

    visualizer = ODEVisualizer(solution)

    stats = visualizer.summary_stats()

    # metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Peak Waiting Room",
        stats["peak_waiting"]
    )

    col2.metric(
        "Peak Doctor Load",
        stats["peak_doctor_load"]
    )

    col3.metric(
        "Peak Pharmacy",
        stats["peak_pharmacy"]
    )

    col4.metric(
        "Discharged by End",
        stats["final_discharged"]
    )

    # forecast plot
    st.pyplot(
        visualizer.plot_forecast()
    )

    # translated alerts
    st.subheader("Operational Alerts")

    if stats["peak_waiting"] > 20:
        st.warning(
            "Heavy waiting room congestion expected."
        )

    elif stats["peak_waiting"] > 10:
        st.info(
            "Moderate waiting room load expected."
        )

    else:
        st.success(
            "Clinic flow appears manageable."
        )

    if stats["peak_doctor_load"] > total_doctors:
        st.error(
            "Doctor bottleneck likely."
        )

    # staffing context
    st.caption(
        f"Forecast based on "
        f"{total_doctors} total doctors"
    )


# tab 4: settings
with tab_settings:

    st.subheader("Clinic Configuration")

    settings = (
        settings_service
        .load_settings()
    )

    total_doctors = st.number_input(
        "Total Doctors",
        min_value=1,
        value=settings["total_doctors"]
    )

    doctor_rooms = st.number_input(
        "Doctor Rooms",
        min_value=1,
        value=settings["doctor_rooms"]
    )

    pharmacy_capacity = st.number_input(
        "Pharmacy Capacity",
        min_value=1,
        value=settings["pharmacy_capacity"]
    )

    clinic_open_hour = st.number_input(
        "Clinic Opening Hour",
        min_value=0,
        max_value=23,
        value=settings["clinic_open_hour"]
    )

    clinic_close_hour = st.number_input(
        "Clinic Closing Hour",
        min_value=1,
        max_value=24,
        value=settings["clinic_close_hour"]
    )

    forecast_days = st.number_input(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=settings["forecast_days"]
    )

    if st.button("Save Settings"):

        new_settings = {
            "total_doctors": total_doctors,
            "doctor_rooms": doctor_rooms,
            "pharmacy_capacity": pharmacy_capacity,
            "clinic_open_hour": clinic_open_hour,
            "clinic_close_hour": clinic_close_hour,
            "forecast_days": forecast_days
        }

        settings_service.save_settings(
            new_settings
        )

        st.success(
            "Settings updated."
        )

# tab 5: logs
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