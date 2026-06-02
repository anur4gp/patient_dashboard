import matplotlib.pyplot as plt
import numpy as np


class ArrivalVisualizer:

    def __init__(
        self,
        lambda_function
    ):
        self.lambda_function = (
            lambda_function
        )

    def plot_arrivals(self):

        times = np.arange(
            0,
            1440,
            10
        )

        rates = [
            self.lambda_function(t)
            for t in times
        ]

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(times, rates)

        ax.set_title(
            "Estimated Patient Arrivals"
        )

        ax.set_xlabel(
            "Minute of Day"
        )

        ax.set_ylabel(
            "Arrival Rate"
        )

        return fig