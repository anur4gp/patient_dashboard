import pandas as pd


class ParameterEstimator:

    def __init__(self, sheets):
        self.logs = sheets["ScanLog"].copy()

        self.logs["timestamp"] = pd.to_datetime(
            self.logs["timestamp"]
        )

        self.logs = self.logs.sort_values(
            "timestamp"
        )

    def estimate_arrival_rate(self):

        arrivals = self.logs[
            self.logs["action"] == "SCAN_LOOKUP"
        ].copy()

        arrivals["hour"] = (
            arrivals["timestamp"]
            .dt.floor("1h")
        )

        counts = (
            arrivals
            .groupby("hour")
            .size()
            .reset_index(name="arrivals")
        )

        return counts

    def estimate_pharmacy_probability(self):

        moves = self.logs[
            self.logs["action"] == "MOVE_COMPARTMENT"
        ]

        pharmacy = (
            moves["metadata"]
            .astype(str)
            .str.upper()
            == "PHARMACY"
        ).sum()

        total = len(moves)

        if total == 0:
            return 0.2

        return pharmacy / total