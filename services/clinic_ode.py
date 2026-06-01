import numpy as np
from scipy.integrate import solve_ivp


class ClinicODESystem:

    def __init__(
        self,
        rates,
        arrival_function,
        total_doctors=4,
        pharmacy_probability=0.2
    ):

        self.rates = rates
        self.arrival_function = arrival_function
        self.total_doctors = total_doctors
        self.pharmacy_probability = pharmacy_probability

        self.gamma_W = rates.get("WAITING", 0.05)
        self.gamma_D = rates.get("DOCTOR", 0.03)
        self.gamma_P = rates.get("PHARMACY", 0.08)


        self.p = pharmacy_probability
        self.y = total_doctors

        self.arrival_function = arrival_function

    

    # ode system
    def system(self, t, state):

        I, W, D, P, X = state

        lam = max(
            self.arrival_function(t / 60),
            0
        )

        # nonlinear doctor availability
        available_doctors = max(
            self.total_doctors - D,
            0
        )

        doctor_capacity = min(
            W,
            available_doctors
        )

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

    # solve_ivp solver
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