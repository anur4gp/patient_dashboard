import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline


class ArrivalModel:

    def __init__(self, sheets):

        self.sheets = sheets

        if "ScanLog" not in sheets:
            raise ValueError(
                "ScanLog missing"
            )

        self.logs = sheets[
            "ScanLog"
        ].copy()

    def build_lambda_function(
        self,
        interval_minutes=30
    ):
        """
        Learn time-of-day arrivals
        from historical scans.

        Returns λ(t).
        """

        logs = self.logs.copy()

        # only patient entry scans
        logs = logs[
            logs["action"]
            == "SCAN_LOOKUP"
        ]

        if logs.empty:
            raise ValueError(
                "No scan data available"
            )

        logs["timestamp"] = pd.to_datetime(
            logs["timestamp"]
        )

        # convert timestamps to
        # minute-of-day
        logs["minute_of_day"] = (
            logs["timestamp"].dt.hour * 60
            +
            logs["timestamp"].dt.minute
        )

        # bucket arrivals
        bins = np.arange(
            0,
            1440 + interval_minutes,
            interval_minutes
        )

        counts, edges = np.histogram(
            logs["minute_of_day"],
            bins=bins
        )

        # convert to rate
        # arrivals per minute
        rates = (
            counts
            / interval_minutes
        )

        midpoints = (
            edges[:-1]
            + interval_minutes / 2
        )

        # smooth curve
        spline = CubicSpline(
            midpoints,
            rates,
            bc_type="natural"
        )

        def lambda_function(t):
            """
            t in minutes
            """

            t_mod = t % 1440

            value = spline(t_mod)

            return max(
                float(value),
                0
            )

        return lambda_function