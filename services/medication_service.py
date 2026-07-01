"""
MedicationService
-----------------
Manages a Medication Administration Record (MAR) stored as an append-only CSV.
Each row is one prescription event. Follow-up prescriptions reference their
predecessor via `prior_rx_id`, forming a chain that can be queried per patient.

CSV columns:
    rx_id            - unique ID for this prescription (uuid4 short)
    patient_uuid     - anonymised patient token (e.g. "Patient #3")
    prescribed_at    - ISO timestamp
    drug_name        - free text
    dose             - free text (e.g. "500mg", "10mg/5mL")
    route            - Oral / IV / Topical / IM / Other
    frequency        - free text (e.g. "TID", "PRN", "Once")
    notes            - free text clinical note
    effectiveness    - Pending / Effective / Partial / Ineffective
    price_usd        - float, manually entered
    prior_rx_id      - rx_id of predecessor prescription, or "" for first Rx
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime
from typing import Optional

MAR_PATH = "medication_log.csv"

COLUMNS = [
    "rx_id",
    "patient_uuid",
    "prescribed_at",
    "drug_name",
    "dose",
    "route",
    "frequency",
    "notes",
    "effectiveness",
    "price_usd",
    "prior_rx_id",
]

ROUTES = ["Oral", "IV", "Topical", "IM", "Other"]
EFFECTIVENESS_OPTIONS = ["Pending", "Effective", "Partial", "Ineffective"]


def _short_id() -> str:
    return uuid.uuid4().hex[:8].upper()


class MedicationService:
    def __init__(self, path: str = MAR_PATH):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def prescribe(
        self,
        patient_uuid: str,
        drug_name: str,
        dose: str,
        route: str,
        frequency: str,
        notes: str = "",
        price_usd: float = 0.0,
        prior_rx_id: str = "",
    ) -> str:
        """Add a new prescription row. Returns the new rx_id."""
        rx_id = _short_id()
        row = {
            "rx_id": rx_id,
            "patient_uuid": patient_uuid,
            "prescribed_at": datetime.now().isoformat(),
            "drug_name": drug_name,
            "dose": dose,
            "route": route,
            "frequency": frequency,
            "notes": notes,
            "effectiveness": "Pending",
            "price_usd": round(float(price_usd), 2),
            "prior_rx_id": prior_rx_id,
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)
        return rx_id

    def update_effectiveness(self, rx_id: str, effectiveness: str, notes: Optional[str] = None):
        """Rewrite the effectiveness (and optionally notes) for one row."""
        rows = self._read_all()
        updated = False
        for row in rows:
            if row["rx_id"] == rx_id:
                row["effectiveness"] = effectiveness
                if notes is not None:
                    row["notes"] = notes
                updated = True
                break
        if updated:
            self._write_all(rows)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def _read_all(self) -> list[dict]:
        with open(self.path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def get_for_patient(self, patient_uuid: str) -> list[dict]:
        """All prescriptions for one patient, oldest first."""
        return [r for r in self._read_all() if r["patient_uuid"] == patient_uuid]

    def get_chain(self, rx_id: str) -> list[dict]:
        """
        Walk the chain from the given rx_id back to the root,
        then return the full forward chain from root.
        """
        all_rows = self._read_all()
        by_id = {r["rx_id"]: r for r in all_rows}

        # walk back to root
        root_id = rx_id
        visited = set()
        while True:
            row = by_id.get(root_id)
            if not row or not row["prior_rx_id"] or root_id in visited:
                break
            visited.add(root_id)
            root_id = row["prior_rx_id"]

        # walk forward from root
        chain = []
        current_id = root_id
        while current_id:
            row = by_id.get(current_id)
            if not row:
                break
            chain.append(row)
            # find the follow-up that points to current_id
            next_rows = [r for r in all_rows if r["prior_rx_id"] == current_id]
            current_id = next_rows[0]["rx_id"] if next_rows else ""

        return chain

    def get_csv_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _write_all(self, rows: list[dict]):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)