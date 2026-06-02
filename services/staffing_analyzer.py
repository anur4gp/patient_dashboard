import numpy as np


class StaffingAnalyzer:

    def __init__(
        self,
        simulations,
        total_doctors
    ):
        self.simulations = simulations
        self.total_doctors = total_doctors

    def congestion_probability(
        self,
        threshold=15
    ):
        waiting = self.simulations[:, 1, :]

        congestion = (
            waiting > threshold
        )

        return congestion.mean(axis=0)

    def peak_congestion_window(
        self,
        threshold=0.7
    ):
        probabilities = (
            self.congestion_probability()
        )

        overloaded = np.where(
            probabilities > threshold
        )[0]

        if len(overloaded) == 0:
            return None

        return (
            overloaded[0],
            overloaded[-1]
        )

    def recommended_doctors(
        self
    ):
        doctor_load = (
            self.simulations[:, 2, :]
        )

        peak_load = np.max(
            doctor_load
        )

        recommended = int(
            np.ceil(
                peak_load / 0.8
            )
        )

        extra_needed = max(
            0,
            recommended
            - self.total_doctors
        )

        return {
            "recommended": recommended,
            "additional_needed": extra_needed
        }