import random
import uuid
from datetime import datetime, timedelta
import pandas as pd


class ClinicSimulator:

    def __init__(
        self,
        n_doctors=4,
        pharmacy_probability=0.65,
        doctor_return_probability=0.10
    ):

        self.n_doctors = n_doctors
        self.pharmacy_probability = pharmacy_probability
        self.doctor_return_probability = doctor_return_probability

    # -------------------------
    # RANDOM DWELL TIMES
    # -------------------------
    def sample_waiting_time(self):
        return max(2, int(random.gauss(12, 5)))

    def sample_doctor_time(self):
        return max(5, int(random.gauss(22, 10)))

    def sample_pharmacy_time(self):
        return max(2, int(random.gauss(8, 4)))

    # -------------------------
    # MAIN SIMULATION
    # -------------------------
    def simulate_day(
        self,
        n_patients=200,
        start_hour=8,
        end_hour=17
    ):

        events = []

        clinic_start = datetime.now().replace(
            hour=start_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        clinic_minutes = (end_hour - start_hour) * 60

        for _ in range(n_patients):

            patient_id = str(uuid.uuid4())[:8]

            # arrival time
            arrival_offset = random.randint(0, clinic_minutes)

            arrival_time = clinic_start + timedelta(
                minutes=arrival_offset
            )

            # -------------------------
            # INTAKE / LOOKUP
            # -------------------------
            events.append({
                "patient_uuid": patient_id,
                "timestamp": arrival_time,
                "action": "SCAN_LOOKUP",
                "source": "simulator",
                "metadata": "INTAKE"
            })

            # -------------------------
            # WAITING
            # -------------------------
            waiting_start = arrival_time + timedelta(
                minutes=random.randint(1, 5)
            )

            events.append({
                "patient_uuid": patient_id,
                "timestamp": waiting_start,
                "action": "MOVE_COMPARTMENT",
                "source": "simulator",
                "metadata": "WAITING"
            })

            # -------------------------
            # DOCTOR
            # -------------------------
            doctor_start = waiting_start + timedelta(
                minutes=self.sample_waiting_time()
            )

            events.append({
                "patient_uuid": patient_id,
                "timestamp": doctor_start,
                "action": "MOVE_COMPARTMENT",
                "source": "simulator",
                "metadata": "DOCTOR"
            })

            doctor_end = doctor_start + timedelta(
                minutes=self.sample_doctor_time()
            )

            # -------------------------
            # PHARMACY OR DISCHARGE
            # -------------------------
            if random.random() < self.pharmacy_probability:

                pharmacy_start = doctor_end

                events.append({
                    "patient_uuid": patient_id,
                    "timestamp": pharmacy_start,
                    "action": "MOVE_COMPARTMENT",
                    "source": "simulator",
                    "metadata": "PHARMACY"
                })

                pharmacy_end = pharmacy_start + timedelta(
                    minutes=self.sample_pharmacy_time()
                )

                # rare doctor return
                if random.random() < self.doctor_return_probability:

                    events.append({
                        "patient_uuid": patient_id,
                        "timestamp": pharmacy_end,
                        "action": "MOVE_COMPARTMENT",
                        "source": "simulator",
                        "metadata": "DOCTOR"
                    })

                    pharmacy_end += timedelta(
                        minutes=self.sample_doctor_time()
                    )

                discharge_time = pharmacy_end

            else:
                discharge_time = doctor_end

            # -------------------------
            # DISCHARGE
            # -------------------------
            events.append({
                "patient_uuid": patient_id,
                "timestamp": discharge_time,
                "action": "MOVE_COMPARTMENT",
                "source": "simulator",
                "metadata": "DISCHARGE"
            })

        logs = pd.DataFrame(events)

        logs = logs.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        return logs
    
    def inject_into_workbook(
        self,
        sheets,
        n_patients=1000
    ):

        simulated_logs = self.simulate_day(
            n_patients=n_patients
        )

        # -------------------------
        # APPEND TO SCANLOG
        # -------------------------
        if "ScanLog" in sheets:

            existing_logs = sheets["ScanLog"].copy()

            # normalize timestamps
            existing_logs["timestamp"] = pd.to_datetime(
                existing_logs["timestamp"],
                errors="coerce"
            )

            simulated_logs["timestamp"] = pd.to_datetime(
                simulated_logs["timestamp"],
                errors="coerce"
            )

            combined_logs = pd.concat(
                [existing_logs, simulated_logs],
                ignore_index=True
            )

        else:
            combined_logs = simulated_logs.copy()

        # normalize again after concat
        combined_logs["timestamp"] = pd.to_datetime(
            combined_logs["timestamp"],
            errors="coerce"
        )

        combined_logs = combined_logs.sort_values(
            "timestamp"
        )

        sheets["ScanLog"] = combined_logs

        # -------------------------
        # UPDATE PATIENT LOCATIONS
        # -------------------------
        movement_logs = combined_logs[
            combined_logs["action"] == "MOVE_COMPARTMENT"
        ].copy()

        latest_locations = (
            movement_logs
            .sort_values("timestamp")
            .groupby("patient_uuid")
            .last()
            .reset_index()
        )

        patient_rows = []

        for _, row in latest_locations.iterrows():

            patient_rows.append({
                "patient_uuid": row["patient_uuid"],
                "first_name": "Sim",
                "last_name": f"Patient_{str(row['patient_uuid'])[:4]}",
                "current_location": row.get(
                    "metadata",
                    "INTAKE"
                )
            })

        sheets["Patients"] = pd.DataFrame(patient_rows)

        return sheets