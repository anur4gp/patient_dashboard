import pandas as pd
import numpy as np

class ModelBuilder:

    def __init__(self, scanlog):
        self.scanlog = scanlog.copy()

        self.scanlog["timestamp"] = pd.to_datetime(
            self.scanlog["timestamp"]
        )

    def estimate_hourly_arrivals(self):

        arrivals = self.scanlog[
            self.scanlog["action"] == "SCAN_LOOKUP"
        ].copy()

        arrivals["hour"] = arrivals["timestamp"].dt.hour

        hourly_lambda = (
            arrivals.groupby("hour")
            .size()
            .reset_index(name="arrivals")
        )

        return hourly_lambda

    def estimate_transition_probabilities(self):

        logs = self.scanlog.sort_values(
            ["patient_uuid", "timestamp"]
        )

        transitions = []

        for patient, group in logs.groupby("patient_uuid"):

            locations = group["metadata"].dropna().tolist()

            for i in range(len(locations)-1):

                transitions.append(
                    (locations[i], locations[i+1])
                )

        transitions_df = pd.DataFrame(
            transitions,
            columns=["from", "to"]
        )

        probs = (
            transitions_df
            .value_counts(normalize=True)
            .reset_index(name="probability")
        )

        return probs