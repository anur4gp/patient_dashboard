import pandas as pd
from datetime import datetime

class ScanService:
    def __init__(self, backend):
        self.backend = backend

    def log_scan(self, sheets, patient_uuid, action="SCAN_LOOKUP", source="streamlit", metadata=None):

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "patient_uuid": patient_uuid,
            "action": action,
            "source": source,
            "metadata": str(metadata) if metadata else ""
        }

        # ensure ScanLog exists
        if "ScanLog" not in sheets:
            sheets["ScanLog"] = pd.DataFrame(columns=log_entry.keys())

        sheets["ScanLog"] = pd.concat(
            [sheets["ScanLog"], pd.DataFrame([log_entry])],
            ignore_index=True
        )

        return sheets