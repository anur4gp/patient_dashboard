import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import random


class ArrivalModel:

    def __init__(self, sheets):

        self.logs = sheets["ScanLog"].copy()

        self.logs["timestamp"] = pd.to_datetime(
            self.logs["timestamp"]
        )

    def _daily_arrival_curves(self):

        arrivals = self.logs[
            self.logs["action"] == "SCAN_LOOKUP"
        ].copy()

        arrivals["date"] = (
            arrivals["timestamp"]
            .dt.date
        )

        daily_curves = []

        for _, group in arrivals.groupby("date"):

            group["minute"] = (
                group["timestamp"].dt.hour * 60
                + group["timestamp"].dt.minute
            )

            bins = np.arange(
                0,
                24 * 60 + 60,
                60
            )

            counts, edges = np.histogram(
                group["minute"],
                bins=bins
            )

            centers = (
                edges[:-1]
                + edges[1:]
            ) / 2

            spline = CubicSpline(
                centers,
                counts,
                extrapolate=True
            )

            daily_curves.append(
                spline
            )

        return daily_curves

    def build_lambda_function(
        self,
        stochastic=True
    ):

        daily_curves = (
            self._daily_arrival_curves()
        )

        if len(daily_curves) == 0:

            return lambda t: 1

        if stochastic:

            chosen_curve = random.choice(
                daily_curves
            )

        else:

            def mean_curve(t):

                vals = [
                    curve(t)
                    for curve in daily_curves
                ]

                return np.mean(vals)

            chosen_curve = mean_curve

        def lambda_function(t):

            value = chosen_curve(t)

            return max(
                0,
                float(value)
            )

        return lambda_function