import csv
import os
from datetime import datetime

EVENT_LOG_PATH = "event_log.csv"
COLUMNS = ["timestamp", "from_compartment", "to_compartment"]


class EventLogService:
    def __init__(self, path: str = EVENT_LOG_PATH):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        """Create the CSV with headers if it doesn't exist yet."""
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()

    def log_move(self, from_compartment: str, to_compartment: str):
        """Append a single compartment transition event."""
        row = {
            "timestamp": datetime.now().isoformat(),
            "from_compartment": from_compartment,
            "to_compartment": to_compartment,
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)

    def read_log(self) -> list[dict]:
        """Return all logged events as a list of dicts."""
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_csv_bytes(self) -> bytes:
        """Return raw CSV bytes for Streamlit download."""
        with open(self.path, "rb") as f:
            return f.read()