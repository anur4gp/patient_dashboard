import pandas as pd
from datetime import datetime

class PatientService:
    def __init__(self, backend):
        self.backend = backend

    def get_patients(self):
        sheets = self.backend.read_patients()

        if "Patients" not in sheets:
            raise Exception("Missing Patients sheet")

        df = sheets["Patients"].copy()

        if "patient_uuid" not in df.columns:
            raise Exception("Missing patient_uuid column")

        if "current_location" not in df.columns:
            df["current_location"] = "INTAKE"

        sheets["Patients"] = df

        return df, sheets

    def move_patient(self, sheets, uuid, new_location):
        df = sheets["Patients"].copy()

        df.loc[df["patient_uuid"] == uuid, "current_location"] = new_location
        df.loc[df["patient_uuid"] == uuid, "last_updated"] = datetime.now().isoformat()

        sheets["Patients"] = df
        return sheets
    
    def update_patient(self, sheets, patient_uuid, updated_fields):

        df = sheets["Patients"].copy()

        for column, value in updated_fields.items():
            if column in df.columns:
                df.loc[
                    df["patient_uuid"] == patient_uuid,
                    column
                ] = value

        df.loc[
            df["patient_uuid"] == patient_uuid,
            "last_updated"
        ] = datetime.now().isoformat()

        sheets["Patients"] = df

        return sheets
    
    def add_patient(self, sheets: dict, new_row: dict) -> dict:
        """Append a new patient row to the Patients sheet."""
        import pandas as pd
        new_df  = pd.DataFrame([new_row])
        pat_key = next(k for k in sheets if "patient" in k.lower())
        sheets[pat_key] = pd.concat(
            [sheets[pat_key], new_df], ignore_index=True
        )
        return sheets