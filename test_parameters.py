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