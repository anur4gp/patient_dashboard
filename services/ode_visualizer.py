import pandas as pd
import matplotlib.pyplot as plt


class ODEVisualizer:

    def __init__(self, solution):

        self.solution = solution

    def build_dataframe(self):

        t = self.solution.t

        df = pd.DataFrame({
            "time_minutes": t,
            "intake": self.solution.y[0],
            "waiting": self.solution.y[1],
            "doctor": self.solution.y[2],
            "pharmacy": self.solution.y[3],
            "discharged": self.solution.y[4]
        })

        return df

    def plot_forecast(self):

        df = self.build_dataframe()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            df["time_minutes"],
            df["waiting"],
            label="Waiting Room"
        )

        ax.plot(
            df["time_minutes"],
            df["doctor"],
            label="Doctor Load"
        )

        ax.plot(
            df["time_minutes"],
            df["pharmacy"],
            label="Pharmacy"
        )

        ax.set_xlabel("Minutes")
        ax.set_ylabel("Patients")

        ax.legend()

        return fig

    def summary_stats(self):

        df = self.build_dataframe()

        return {
            "peak_waiting":
                round(df["waiting"].max(), 1),

            "peak_doctor_load":
                round(df["doctor"].max(), 1),

            "peak_pharmacy":
                round(df["pharmacy"].max(), 1),

            "final_discharged":
                round(df["discharged"].iloc[-1], 1)
        }