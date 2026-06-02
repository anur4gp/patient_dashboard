import matplotlib.pyplot as plt
import numpy as np


class ODEVisualizer:

    def __init__(self, solution):
        self.solution = solution

    def plot_forecast(self):

        fig, ax = plt.subplots(figsize=(10, 5))

        labels = [
            "Intake",
            "Waiting Room",
            "Doctor",
            "Pharmacy",
            "Discharged"
        ]

        for i, label in enumerate(labels):

            ax.plot(
                self.solution.t,
                self.solution.y[i],
                label=label
            )

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Patients")
        ax.set_title("Clinic Flow Forecast")
        ax.legend()

        return fig

    def plot_multi_day_forecast(
        self,
        summary,
        compartment_index=1
    ):
        """
        Plot median + uncertainty band
        for one compartment.
        """

        median = summary["median"]
        lower = summary["lower"]
        upper = summary["upper"]

        time_points = np.arange(
            median.shape[1]
        )

        labels = [
            "Intake",
            "Waiting Room",
            "Doctor",
            "Pharmacy",
            "Discharged"
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            time_points,
            median[compartment_index],
            label="Expected"
        )

        ax.fill_between(
            time_points,
            lower[compartment_index],
            upper[compartment_index],
            alpha=0.3,
            label="95% Forecast Range"
        )

        ax.set_title(
            f"{labels[compartment_index]} Forecast"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Patients"
        )

        ax.legend()

        return fig

    def summary_stats(self):

        waiting = self.solution.y[1]
        doctor = self.solution.y[2]
        pharmacy = self.solution.y[3]
        discharged = self.solution.y[4]

        return {
            "peak_waiting": round(
                np.max(waiting), 1
            ),

            "peak_doctor_load": round(
                np.max(doctor), 1
            ),

            "peak_pharmacy": round(
                np.max(pharmacy), 1
            ),

            "final_discharged": round(
                discharged[-1], 1
            )
        }