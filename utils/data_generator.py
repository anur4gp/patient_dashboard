import pandas as pd
import random
import uuid
from faker import Faker

fake = Faker()

meds = ["Lisinopril", "Metformin", "Atorvastatin", "Albuterol"]
conditions = ["HTN", "Diabetes", "Asthma", "High Cholesterol"]

def generate_patients(n=100):
    data = []

    for _ in range(n):
        data.append({
            "patient_uuid": str(uuid.uuid4())[:8],
            "mrn": f"MRN-{random.randint(10000,99999)}",
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "dob": fake.date_of_birth(),
            "phone": fake.phone_number(),
            "conditions": ", ".join(random.sample(conditions, 2)),
            "medications": ", ".join(random.sample(meds, 2)),
            "provider": fake.name(),
            "status": "Checked In"
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_patients()
    df.to_excel("data/mock_patients.xlsx", index=False)
    print("Created mock_patients.xlsx")