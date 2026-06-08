"""
compartment_ode.py
──────────────────
Defines the 3-state ODE system and solves it with a 5th-order
Runge-Kutta integrator (Dormand-Prince / RK45 via scipy, which is
a true adaptive RK5 — better step control than a fixed-step RK4).

State vector:  y = [W, D, P]
    W = patients in Waiting
    D = patients with Doctor
    P = patients in Pharmacy

Equations:
    dW/dt = λ(t) − (α + μ_lwbs)·W
    dD/dt = α·W − (β + γ)·D
    dP/dt = β·D − δ·P

Discharge is recovered post-hoc:
    Dis(t) = ∫λ dt − W(t) − D(t) − P(t)

All rates are in units of patients/hour.
Time axis is hours from midnight of the simulation day (0 → 24).
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import List, Optional
from services.compartment_params import ModelParams


@dataclass
class ODESolution:
    """Container returned by the solver."""
    t:          np.ndarray          # time points (hours from midnight)
    W:          np.ndarray          # Waiting occupancy
    D:          np.ndarray          # Doctor occupancy
    P:          np.ndarray          # Pharmacy occupancy
    Discharge:  np.ndarray          # cumulative discharges (conservation)
    lambda_t:   np.ndarray          # λ(t) evaluated at each time point
    success:    bool = True
    message:    str  = ""

    # Derived metrics (populated by compute_metrics)
    peak_waiting:      float = 0.0
    peak_waiting_hour: float = 0.0
    bottleneck:        str   = ""
    mean_system_time:  float = 0.0  # hours


def build_lambda_fn(lambda_hourly: List[float]):
    """
    Returns a callable λ(t) that interpolates the 24-point hourly vector.
    Uses linear interpolation so the function is continuous — important
    for the adaptive step-size control in RK5.
    """
    hours = np.arange(24, dtype=float)
    rates = np.array(lambda_hourly, dtype=float)

    def lambda_fn(t: float) -> float:
        # Wrap t into [0, 24) so the model can run past midnight if needed
        t_mod = t % 24
        return float(np.interp(t_mod, hours, rates))

    return lambda_fn


def make_ode_system(params: ModelParams):
    """
    Closure that captures ModelParams and returns the RHS function
    f(t, y) expected by scipy.integrate.solve_ivp.

    y[0] = W,  y[1] = D,  y[2] = P
    """
    lambda_fn = build_lambda_fn(params.lambda_hourly)
    alpha    = params.alpha
    beta     = params.beta
    gamma    = params.gamma
    delta    = params.delta
    mu_lwbs  = params.mu_lwbs

    def f(t: float, y: np.ndarray) -> np.ndarray:
        W, D, P = y[0], y[1], y[2]

        # Floor at 0 — compartments can't go negative
        # (the solver may push slightly negative during stiff steps)
        W = max(W, 0.0)
        D = max(D, 0.0)
        P = max(P, 0.0)

        lam = lambda_fn(t)

        dW = lam - (alpha + mu_lwbs) * W
        dD = alpha * W - (beta + gamma) * D
        dP = beta * D - delta * P

        return np.array([dW, dD, dP])

    return f, lambda_fn


def solve(
    params:     ModelParams,
    y0:         Optional[List[float]] = None,
    t_span:     tuple = (0.0, 24.0),
    n_points:   int   = 289,           # every 5 minutes across 24 h
    rtol:       float = 1e-6,
    atol:       float = 1e-8,
) -> ODESolution:
    """
    Solve the ODE system using RK45 (Dormand-Prince — adaptive RK5).

    Parameters
    ----------
    params    : ModelParams from compartment_params.extract_params()
    y0        : Initial state [W0, D0, P0]. Defaults to current observed
                occupancy or [0, 0, 0] if not provided.
    t_span    : (t_start, t_end) in hours from midnight
    n_points  : number of output time points (doesn't affect accuracy,
                only output resolution)
    rtol/atol : solver tolerances — tighter = more accurate but slower.
                These defaults are appropriate for patient-count scales.

    Returns
    -------
    ODESolution with all trajectories and derived metrics.
    """
    if y0 is None:
        y0 = [0.0, 0.0, 0.0]

    y0 = np.array(y0, dtype=float)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    f, lambda_fn = make_ode_system(params)

    result = solve_ivp(
        f,
        t_span,
        y0,
        method="RK45",       # Dormand-Prince adaptive RK5
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not result.success:
        return ODESolution(
            t=t_eval,
            W=np.zeros(n_points),
            D=np.zeros(n_points),
            P=np.zeros(n_points),
            Discharge=np.zeros(n_points),
            lambda_t=np.zeros(n_points),
            success=False,
            message=result.message,
        )

    W = np.maximum(result.y[0], 0)
    D = np.maximum(result.y[1], 0)
    P = np.maximum(result.y[2], 0)

    # λ(t) at each output point (for plotting)
    lambda_t = np.array([lambda_fn(t) for t in t_eval])

    # Cumulative arrivals via trapezoidal integration of λ(t)
    cumulative_arrivals = np.cumsum(lambda_t) * (t_eval[1] - t_eval[0])

    # Conservation: Discharge = total arrived − still in system
    Discharge = np.maximum(cumulative_arrivals - W - D - P, 0)

    sol = ODESolution(
        t=t_eval,
        W=W, D=D, P=P,
        Discharge=Discharge,
        lambda_t=lambda_t,
        success=True,
        message="OK",
    )

    _compute_metrics(sol, params)
    return sol


# ── Post-processing ───────────────────────────────────────────────────────

def _compute_metrics(sol: ODESolution, params: ModelParams):
    """Populate derived metrics on the solution object in-place."""

    # Peak waiting room load
    sol.peak_waiting      = float(np.max(sol.W))
    sol.peak_waiting_hour = float(sol.t[np.argmax(sol.W)])

    # Identify bottleneck: whichever compartment has highest peak-to-rate ratio
    # (high occupancy relative to its outflow rate → longest queue)
    peaks = {
        "Waiting":  sol.peak_waiting        / max(params.alpha, 1e-9),
        "Doctor":   float(np.max(sol.D))    / max(params.beta + params.gamma, 1e-9),
        "Pharmacy": float(np.max(sol.P))    / max(params.delta, 1e-9),
    }
    sol.bottleneck = max(peaks, key=peaks.get)

    # Mean time in system: Little's Law approximation
    # L = λ̄ · W  →  W̄ = L / λ̄
    mean_lambda = float(np.mean(sol.lambda_t))
    if mean_lambda > 0:
        mean_total_occupancy = float(np.mean(sol.W + sol.D + sol.P))
        sol.mean_system_time = mean_total_occupancy / mean_lambda  # hours
    else:
        sol.mean_system_time = 0.0


def get_current_occupancy(log_df, compartments=("WAITING", "DOCTOR", "PHARMACY")):
    """
    Derive current real-world occupancy from the ScanLog to use as y0.
    For each patient, find their most recent MOVE_COMPARTMENT destination
    and count how many are currently in each tracked compartment.

    Returns [W_now, D_now, P_now] as floats.
    """
    if log_df is None or log_df.empty:
        return [0.0, 0.0, 0.0]

    df = log_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    moves = df[df["action"] == "MOVE_COMPARTMENT"].copy()
    moves["metadata"] = moves["metadata"].astype(str).str.upper().str.strip()

    if moves.empty:
        return [0.0, 0.0, 0.0]

    # Latest location per patient
    latest = moves.sort_values("timestamp").groupby("patient_uuid").last()
    counts = latest["metadata"].value_counts()

    return [
        float(counts.get("WAITING",  0)),
        float(counts.get("DOCTOR",   0)),
        float(counts.get("PHARMACY", 0)),
    ]


# pandas needed in get_current_occupancy — import at module level
import pandas as pd
