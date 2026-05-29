from services.clinic_simulator import ClinicSimulator
from services.excel_backend import ExcelBackend

backend = ExcelBackend(
    "data/mock_patients.xlsx"
)

sheets = backend.read_patients()

sim = ClinicSimulator()

sheets = sim.inject_into_workbook(
    sheets,
    n_patients=1000
)

backend.write_patients(sheets)

print("Simulation written to workbook.")