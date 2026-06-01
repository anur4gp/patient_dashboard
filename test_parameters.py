from services.excel_backend import ExcelBackend
from services.parameter_estimator import ParameterEstimator

backend = ExcelBackend(
    "data/mock_patients.xlsx"
)

sheets = backend.read_patients()

estimator = ParameterEstimator(sheets)

print("\nARRIVAL RATES:")
print(estimator.estimate_arrival_rate())

print("\nPHARMACY PROBABILITY:")
print(estimator.estimate_pharmacy_probability())

print("\nDWELL TIMES:")
dwell = estimator.estimate_dwell_times()

for k, v in dwell.items():
    print(k, len(v), "samples")
    print(v[:10])

print("\nTRANSITION RATES:")
rates = estimator.estimate_transition_rates()

for k, v in rates.items():
    print(k, "=", v)