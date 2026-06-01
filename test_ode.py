from services.excel_backend import ExcelBackend
from services.parameter_estimator import ParameterEstimator
from services.clinic_ode import ClinicODESystem

backend = ExcelBackend(
    "data/mock_patients.xlsx"
)

sheets = backend.read_patients()

estimator = ParameterEstimator(sheets)

rates = estimator.estimate_transition_rates()

p = estimator.estimate_pharmacy_probability()

print("\nRATES:")
print(rates)

print("\nPHARMACY PROBABILITY:")
print(p)

ode = ClinicODESystem(
    rates=rates,
    pharmacy_probability=p,
    doctors=4
)

solution = ode.solve(
    initial_state=[
        0,  # Intake
        0,  # Waiting
        0,  # Doctor
        0,  # Pharmacy
        0   # Discharged
    ],
    t_end=600
)

print("\nFINAL STATE:")
print(solution.y[:, -1])