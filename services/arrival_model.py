import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline


class ArrivalModel:

    def __init__(self, sheets):

        self.sheets = sheets

    def build_lambda_function(
        self,
        interval_minutes=60
    ):

        if "ScanLog" not in self.sheets:
            raise Exception("Missing ScanLog")

        logs = self.sheets["ScanLog"].copy()

        # only arrival events
        arrivals = logs[
            logs["action"] == "SCAN_LOOKUP"
        ].copy()

        if len(arrivals) == 0:
            raise Exception(
                "No scan lookup data found"
            )

        arrivals["timestamp"] = pd.to_datetime(
            arrivals["timestamp"]
        )

        arrivals["hour_decimal"] = (
            arrivals["timestamp"].dt.hour
            +
            arrivals["timestamp"].dt.minute / 60
        )

        # bin counts
        bins = np.arange(0, 24, interval_minutes / 60)

        counts, edges = np.histogram(
            arrivals["hour_decimal"],
            bins=bins
        )

        centers = (
            edges[:-1]
            + edges[1:]
        ) / 2

        # smooth intensity
        spline = CubicSpline(
            centers,
            counts,
            bc_type="natural"
        )

        return spline