import numpy as np
import pandas as pd

from services.arrival_model import ArrivalModel
from services.clinic_ode import ClinicODESystem


class MultiDaySimulator:

    def __init__(
        self,
        sheets,
        rates,
        pharmacy_probability,
        total_doctors
    ):

        self.sheets = sheets
        self.rates = rates
        self.p = pharmacy_probability
        self.total_doctors = total_doctors

    def simulate_days(
        self,
        n_days=7,
        t_end=600,
        points=500
    ):

        simulations = []

        for _ in range(n_days):

            arrival_model = ArrivalModel(
                self.sheets
            )

            lambda_function = (
                arrival_model
                .build_lambda_function(
                    stochastic=True
                )
            )

            ode = ClinicODESystem(
                rates=self.rates,
                arrival_function=lambda_function,
                pharmacy_probability=self.p,
                total_doctors=self.total_doctors
            )

            solution = ode.solve(
                initial_state=[0,0,0,0,0],
                t_end=t_end,
                points=points
            )

            simulations.append(
                solution.y
            )

        return np.array(
            simulations
        )

    def summarize(self, sims):

        mean_curve = sims.mean(axis=0)

        lower = np.percentile(
            sims,
            10,
            axis=0
        )

        upper = np.percentile(
            sims,
            90,
            axis=0
        )

        return {
            "mean": mean_curve,
            "lower": lower,
            "upper": upper
        }