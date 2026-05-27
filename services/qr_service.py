import qrcode
import os

def generate_qr(patient_uuid):
    os.makedirs("qr_codes", exist_ok=True)

    img = qrcode.make(patient_uuid)
    path = f"qr_codes/{patient_uuid}.png"
    img.save(path)

    return path