from services.clinic_simulator import ClinicSimulator

sim = ClinicSimulator()

logs = sim.simulate_day(
    n_patients=200
)

print(logs.head(20))
print(logs.tail(20))