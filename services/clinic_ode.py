import numpy as np
from scipy.integrate import solve_ivp


class ClinicODESystem:

    def __init__(
        self,
        rates,
        pharmacy_probability=0.2,
        doctors=3
    ):

        self.gamma_W = rates.get("WAITING", 0.05)
        self.gamma_D = rates.get("DOCTOR", 0.03)
        self.gamma_P = rates.get("PHARMACY", 0.08)

        self.p = pharmacy_probability
        self.y = doctors

    # -----------------------------------
    # Arrival function λ(t)
    # -----------------------------------
    def arrival_rate(self, t):

        # simple time-dependent arrivals
        # t measured in minutes

        hour = t / 60

        # morning rush
        if 8 <= hour <= 11:
            return 4

        # lunch slowdown
        elif 11 < hour <= 13:
            return 2

        # afternoon moderate
        elif 13 < hour <= 17:
            return 3

        return 0.5

    # -----------------------------------
    # ODE SYSTEM
    # -----------------------------------
    def system(self, t, state):

        I, W, D, P, X = state

        lam = self.arrival_rate(t)

        # nonlinear doctor availability
        available_doctors = max(self.y - D, 0)

        flow_to_doctor = min(
            W,
            available_doctors
        )

        dI = lam - self.gamma_W * I

        dW = (
            self.gamma_W * I
            - flow_to_doctor
        )

        dD = (
            flow_to_doctor
            - self.gamma_D * D
        )

        dP = (
            self.p * self.gamma_D * D
            - self.gamma_P * P
        )

        dX = (
            (1 - self.p)
            * self.gamma_D
            * D
            +
            self.gamma_P * P
        )

        return [
            dI,
            dW,
            dD,
            dP,
            dX
        ]

    # -----------------------------------
    # SOLVER
    # -----------------------------------
    def solve(
        self,
        initial_state,
        t_end=600,
        points=1000
    ):

        t_eval = np.linspace(
            0,
            t_end,
            points
        )

        sol = solve_ivp(
            self.system,
            [0, t_end],
            initial_state,
            t_eval=t_eval,
            method="RK45"
        )

        return sol