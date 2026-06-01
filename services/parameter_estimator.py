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
    
    def estimate_dwell_times(self):

        moves = self.logs[
            self.logs["action"] == "MOVE_COMPARTMENT"
        ].copy()

        moves["metadata"] = (
            moves["metadata"]
            .astype(str)
            .str.upper()
        )

        dwell = {
            "WAITING": [],
            "DOCTOR": [],
            "PHARMACY": []
        }

        for patient_id in moves["patient_uuid"].unique():

            patient_moves = (
                moves[
                    moves["patient_uuid"]
                    == patient_id
                ]
                .sort_values("timestamp")
            )

            patient_moves = patient_moves.reset_index(
                drop=True
            )

            for i in range(len(patient_moves)-1):

                current = patient_moves.iloc[i]
                nxt = patient_moves.iloc[i+1]

                compartment = current["metadata"]

                delta = (
                    nxt["timestamp"]
                    - current["timestamp"]
                ).total_seconds() / 60

                if compartment in dwell:
                    dwell[compartment].append(delta)

        return dwell
    
    def estimate_transition_rates(self):

        dwell = self.estimate_dwell_times()

        rates = {}

        for compartment, times in dwell.items():

            if len(times) == 0:
                rates[compartment] = 0
                continue

            mean_time = sum(times) / len(times)

            # convert minutes → rate per minute
            rates[compartment] = 1 / mean_time

        return rates