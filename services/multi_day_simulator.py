import numpy as np

from services.clinic_ode import ClinicODESystem
from services.arrival_model import ArrivalModel


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
        self.pharmacy_probability = pharmacy_probability
        self.total_doctors = total_doctors

    def perturb_rates(self):
        """
        Add randomness to transition rates
        to model day-to-day uncertainty.
        """

        noisy_rates = {}

        for key, value in self.rates.items():

            noise = np.random.lognormal(
                mean=0,
                sigma=0.2
            )

            noisy_rates[key] = max(
                value * noise,
                1e-5
            )

        return noisy_rates

    def simulate_days(
        self,
        n_days=7,
        t_end=600
    ):

        arrival_model = ArrivalModel(
            self.sheets
        )

        lambda_function = (
            arrival_model
            .build_lambda_function()
        )

        simulations = []

        for _ in range(n_days):

            perturbed_rates = (
                self.perturb_rates()
            )

            ode = ClinicODESystem(
                rates=perturbed_rates,
                arrival_function=lambda_function,
                pharmacy_probability=self.pharmacy_probability,
                total_doctors=self.total_doctors
            )

            solution = ode.solve(
                initial_state=[0, 0, 0, 0, 0],
                t_end=t_end
            )

            simulations.append(
                solution.y
            )

        return np.array(simulations)

    def summarize(
        self,
        simulations
    ):
        """
        Build uncertainty bands
        from Monte Carlo runs.
        """

        median = np.median(
            simulations,
            axis=0
        )

        lower = np.percentile(
            simulations,
            5,
            axis=0
        )

        upper = np.percentile(
            simulations,
            95,
            axis=0
        )

        return {
            "median": median,
            "lower": lower,
            "upper": upper
        }