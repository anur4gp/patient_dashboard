from services.excel_backend import ExcelBackend
from services.parameter_estimator import ParameterEstimator
from services.clinic_ode import ClinicODESystem
import numpy as np

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

def test_arrival(t):
    """Simple morning-peak arrival curve for testing."""
    return 8.0 * np.exp(-0.5 * ((t - 10.0) / 2.0) ** 2)  # peaks at t=10 hrs

ode = ClinicODESystem(
    rates=rates,
    pharmacy_probability=p,
    total_doctors=4,
    arrival_function=test_arrival
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