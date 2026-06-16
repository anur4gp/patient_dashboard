"""
research/model.py
-----------------
Hybrid stochastic compartment model for patient flow dynamics.

Compartments (matching config.py):
    I  — Intake
    W  — Waiting
    D  — Doctor (Physician)
    P  — Pharmacy
    X  — Discharged

The system is formalized as a Piecewise Deterministic Markov Process (PDMP):
    - Between Poisson arrival events, the state evolves under the deterministic ODE
    - At each arrival event, a single patient is injected into Intake

ODE system:
    dI/dt =  λ(t) - γ_I * I
    dW/dt =  γ_I * I  - α * min(W, y - D)
    dD/dt =  α * min(W, y - D)  - γ_D * D
    dP/dt =  p * γ_D * D  - γ_P * P
    dX/dt =  (1-p) * γ_D * D  + γ_P * P

Parameters
----------
lam    : float  — Poisson arrival rate (patients / min)
gamma_I: float  — Intake → Waiting transition rate
gamma_D: float  — Doctor → (Pharmacy | Discharge) transition rate
gamma_P: float  — Pharmacy → Discharge transition rate
alpha  : float  — Waiting → Doctor transition rate (limited by capacity)
p      : float  — Routing probability Doctor → Pharmacy
y      : float  — Physician capacity (max simultaneous patients with Doctor)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """
    All model parameters in one place.

    Default values are plausible for a small outpatient clinic
    and are used for synthetic experiments when empirical data
    is unavailable.
    """
    lam:     float = 0.5    # arrivals per minute (~30/hr)
    gamma_I: float = 0.2    # avg 5 min in intake
    gamma_D: float = 0.05   # avg 20 min with doctor
    gamma_P: float = 0.1    # avg 10 min in pharmacy
    alpha:   float = 0.15   # waiting-to-doctor transfer rate
    p:       float = 0.4    # 40% routed to pharmacy
    y:       float = 5.0    # physician capacity (slots)

    def to_array(self) -> np.ndarray:
        """Return params as ordered array for use in sensitivity sampling."""
        return np.array([
            self.lam, self.gamma_I, self.gamma_D,
            self.gamma_P, self.alpha, self.p, self.y
        ])

    @staticmethod
    def from_array(arr: np.ndarray) -> "ModelParams":
        """Reconstruct ModelParams from a 7-element array (same order as to_array)."""
        return ModelParams(
            lam=arr[0], gamma_I=arr[1], gamma_D=arr[2],
            gamma_P=arr[3], alpha=arr[4], p=arr[5], y=arr[6]
        )

    @staticmethod
    def param_names() -> list:
        return ["lambda", "gamma_I", "gamma_D", "gamma_P", "alpha", "p", "y"]

    def critical_arrival_rate(self) -> float:
        """
        Analytically derived congestion threshold.

        Below λ_c: waiting compartment stabilizes (unconstrained regime).
        Above λ_c: physician saturates, waiting queue diverges.

        λ_c = y / (1/alpha + 1/gamma_D)
        """
        return self.y / (1.0 / self.alpha + 1.0 / self.gamma_D)


# ---------------------------------------------------------------------------
# Deterministic ODE vector field
# ---------------------------------------------------------------------------

def ode_rhs(t: float, state: np.ndarray, params: ModelParams) -> np.ndarray:
    """
    Right-hand side of the deterministic compartment ODE.

    Parameters
    ----------
    t      : current time (unused for autonomous system; kept for solve_ivp compat)
    state  : [I, W, D, P, X]
    params : ModelParams instance

    Returns
    -------
    dydt   : np.ndarray of shape (5,)
    """
    I, W, D, P, X = state
    p = params

    # Physician capacity constraint — the key nonlinearity
    physician_flow = p.alpha * min(W, max(p.y - D, 0.0))

    dI = p.lam - p.gamma_I * I
    dW = p.gamma_I * I - physician_flow
    dD = physician_flow - p.gamma_D * D
    dP = p.p * p.gamma_D * D - p.gamma_P * P
    dX = (1 - p.p) * p.gamma_D * D + p.gamma_P * P

    return np.array([dI, dW, dD, dP, dX])


def solve_deterministic(
    params: ModelParams,
    T: float = 480.0,
    dt: float = 0.5,
    state0: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the deterministic ODE over [0, T].

    Parameters
    ----------
    params : ModelParams
    T      : simulation horizon (minutes); default 480 = 8-hour clinic day
    dt     : output time step
    state0 : initial state; defaults to all zeros

    Returns
    -------
    t_eval : 1-D array of time points
    sol    : array of shape (5, len(t_eval)) — one row per compartment
    """
    if state0 is None:
        state0 = np.zeros(5)

    t_eval = np.arange(0, T + dt, dt)

    result = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, params),
        t_span=(0, T),
        y0=state0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    return result.t, result.y


# ---------------------------------------------------------------------------
# PDMP simulator (stochastic)
# ---------------------------------------------------------------------------

def simulate_pdmp(
    params: ModelParams,
    T: float = 480.0,
    dt: float = 0.5,
    state0: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one trajectory of the PDMP.

    Between Poisson arrival events the system follows the deterministic ODE.
    At each arrival, a unit jump is applied to the Intake compartment.

    X(t) = X(0) + ∫₀ᵗ F(X(s)) ds + Σₙ Jₙ

    where Jₙ = e₁ (a patient entering Intake).

    Parameters
    ----------
    params : ModelParams
    T      : simulation horizon (minutes)
    dt     : recording time step (ODE is solved between jumps; recorded on grid)
    state0 : initial state; defaults to zeros
    rng    : numpy random Generator for reproducibility

    Returns
    -------
    t_grid : 1-D array of recorded time points
    traj   : array (5, len(t_grid)) — compartment occupancies over time
    """
    if state0 is None:
        state0 = np.zeros(5)
    if rng is None:
        rng = np.random.default_rng()

    t_grid = np.arange(0, T + dt, dt)
    traj = np.zeros((5, len(t_grid)))
    traj[:, 0] = state0.copy()

    state = state0.copy()
    t = 0.0
    grid_idx = 1  # next grid index to fill

    # Pre-draw all inter-arrival times for the horizon
    # Expected arrivals: lam * T; draw 3x for safety
    n_expected = int(params.lam * T * 3) + 50
    inter_arrivals = rng.exponential(scale=1.0 / params.lam, size=n_expected)

    arrival_ptr = 0
    next_arrival = inter_arrivals[arrival_ptr]

    while t < T and arrival_ptr < n_expected:
        # Time of next event
        t_event = min(next_arrival, T)

        # Solve ODE from current state over [t, t_event]
        if t_event > t:
            seg = solve_ivp(
                fun=lambda s, y: ode_rhs(s, y, params),
                t_span=(t, t_event),
                y0=state,
                method="RK45",
                dense_output=True,
                rtol=1e-6,
                atol=1e-8,
            )
            # Fill grid points that fall in [t, t_event]
            while grid_idx < len(t_grid) and t_grid[grid_idx] <= t_event:
                traj[:, grid_idx] = seg.sol(t_grid[grid_idx])
                traj[:, grid_idx] = np.maximum(traj[:, grid_idx], 0)  # non-negativity
                grid_idx += 1

            state = seg.sol(t_event)
            state = np.maximum(state, 0.0)

        t = t_event

        # Apply arrival jump if we haven't hit T
        if t < T:
            state[0] += 1.0  # one patient enters Intake
            arrival_ptr += 1
            if arrival_ptr < n_expected:
                next_arrival = t + inter_arrivals[arrival_ptr]

    # Fill any remaining grid points with final state
    while grid_idx < len(traj.shape[1] if hasattr(traj, 'shape') else len(traj[0])):
        traj[:, grid_idx] = state
        grid_idx += 1

    # Safety fill for last grid point
    traj[:, -1] = state

    return t_grid, traj


def run_monte_carlo(
    params: ModelParams,
    n_trajectories: int = 200,
    T: float = 480.0,
    dt: float = 0.5,
    state0: Optional[np.ndarray] = None,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo ensemble of PDMP trajectories.

    Parameters
    ----------
    params         : ModelParams
    n_trajectories : number of independent trajectories
    T, dt          : simulation horizon and time step
    state0         : initial state
    seed           : base random seed

    Returns
    -------
    dict with keys:
        't'       : time grid (1-D)
        'mean'    : mean trajectory (5, n_times)
        'std'     : std trajectory  (5, n_times)
        'all'     : all trajectories (n_traj, 5, n_times)
        'steady'  : steady-state values (last 10% of each trajectory), shape (n_traj, 5)
    """
    rng = np.random.default_rng(seed)
    t_grid = np.arange(0, T + dt, dt)
    n_times = len(t_grid)
    all_traj = np.zeros((n_trajectories, 5, n_times))

    for i in range(n_trajectories):
        _, traj = simulate_pdmp(params, T=T, dt=dt, state0=state0,
                                rng=np.random.default_rng(seed + i))
        n = min(traj.shape[1], n_times)
        all_traj[i, :, :n] = traj[:, :n]

    # Steady-state: mean of last 10% of time horizon
    burnin_idx = int(0.9 * n_times)
    steady = all_traj[:, :, burnin_idx:].mean(axis=2)  # (n_traj, 5)

    return {
        "t":      t_grid,
        "mean":   all_traj.mean(axis=0),
        "std":    all_traj.std(axis=0),
        "all":    all_traj,
        "steady": steady,
    }


# ---------------------------------------------------------------------------
# Analytical steady state
# ---------------------------------------------------------------------------

def analytical_steady_state(params: ModelParams) -> dict:
    """
    Compute the deterministic steady state analytically.

    Valid only in the unconstrained regime (λ < λ_c).
    Returns None for congested compartments when λ >= λ_c.

    Returns
    -------
    dict with keys: I*, W*, D*, P*, discharge_rate, lambda_c, regime
    """
    lam_c = params.critical_arrival_rate()
    regime = "unconstrained" if params.lam < lam_c else "congested"

    I_star = params.lam / params.gamma_I
    W_star = params.lam / params.alpha
    D_star = params.lam / params.gamma_D
    P_star = params.p * params.lam / params.gamma_P
    discharge_rate = params.lam  # mass conservation check

    return {
        "I_star":         I_star,
        "W_star":         W_star,
        "D_star":         D_star,
        "P_star":         P_star,
        "discharge_rate": discharge_rate,
        "lambda_c":       lam_c,
        "regime":         regime,
    }