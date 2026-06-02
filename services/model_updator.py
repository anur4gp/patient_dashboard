from services.parameter_estimator import (
    ParameterEstimator
)
from services.arrival_model import (
    ArrivalModel
)


class ModelUpdater:

    def update(self, sheets):

        estimator = (
            ParameterEstimator(sheets)
        )

        rates = (
            estimator
            .estimate_transition_rates()
        )

        pharmacy_prob = (
            estimator
            .estimate_pharmacy_probability()
        )

        arrival_model = (
            ArrivalModel(sheets)
        )

        lambda_function = (
            arrival_model
            .build_lambda_function()
        )

        return {
            "rates": rates,
            "pharmacy_probability":
                pharmacy_prob,
            "lambda":
                lambda_function
        }