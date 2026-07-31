"""
patient_qr_service.py

Generates QR codes that encode an ECW Patient FHIR `id` (the opaque
lookup key), not raw demographic data. Scanning the QR is just a
fast way to hand that id to ecw_client.get_patient_by_id() — the
actual PII is fetched live from ECW at scan time and never stored
in the QR image itself.

This mirrors the same principle your existing QR/MRN system already
uses: the QR is a *reference*, not a *payload*.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import qrcode
from qrcode.constants import ERROR_CORRECT_M


QR_PAYLOAD_VERSION = 1


@dataclass
class QRPayload:
    version: int
    ecw_patient_id: str

    def to_json(self) -> str:
        return json.dumps({"v": self.version, "pid": self.ecw_patient_id})

    @classmethod
    def from_json(cls, raw: str) -> "QRPayload":
        data = json.loads(raw)
        return cls(version=data["v"], ecw_patient_id=data["pid"])


def generate_patient_qr(ecw_patient_id: str) -> bytes:
    """Returns PNG bytes for a QR code encoding the given ECW
    Patient id. Feed this straight into your existing label/print
    service the same way you do for the current MRN-based QR."""
    payload = QRPayload(version=QR_PAYLOAD_VERSION, ecw_patient_id=ecw_patient_id)

    qr = qrcode.QRCode(
        version=1,
        error_correction=ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(payload.to_json())
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    # qrcode's PilImage.save expects the image kind as the second arg (named `kind`),
    # not `format`. Pass it as a keyword to avoid type errors from static checkers.
    img.save(buf, kind="PNG")
    return buf.getvalue()


def decode_scanned_payload(scanned_text: str) -> str:
    """Given the raw text pulled off a scanned QR, return the ECW
    patient id to look up. Raises ValueError on malformed/unknown
    payloads so the scan tab can show a clear error instead of a
    stack trace."""
    try:
        payload = QRPayload.from_json(scanned_text)
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError("Unrecognized QR payload") from exc

    if payload.version != QR_PAYLOAD_VERSION:
        raise ValueError(f"Unsupported QR payload version: {payload.version}")

    return payload.ecw_patient_id
