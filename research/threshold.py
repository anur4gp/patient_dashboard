"""
research/threshold.py
---------------------
Congestion threshold sweep: characterizes the critical arrival rate λ_c.

The analytical result from steady-state analysis gives:

    λ_c = y / (1/α + 1/γ_D)

Below λ_c: W* is finite and stable (unconstrained regime).
Above λ_c: physician capacity saturates, W grows without bound (congested regime).

This module:
    1. Sweeps λ across a range bracketing λ_c
    2. For each λ, runs Monte Carlo and records steady-state W*
    3. Computes the empirical threshold from simulation and compares to analytical
    4. Produces data ready for Figure 2 in the paper

All functions return plain arrays/dicts so they can be passed directly
to plot.py without any coupling between modules.
"""

import numpy as np
from research.model import ModelParams, run_monte_carlo, analytical_steady_state


def lambda_sweep(
    base_params: ModelParams,
    n_lambda: int = 30,
    lambda_range_factor: float = 2.0,
    n_mc: int = 100,
    T: float = 480.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Sweep arrival rate λ across [λ_min, λ_max] and record steady-state behavior.

    λ_min = λ_c / lambda_range_factor
    λ_max = λ_c * lambda_range_factor

    This brackets the theoretical threshold symmetrically on a log scale,
    ensuring good coverage of both regimes.

    Parameters
    ----------
    base_params          : ModelParams; all params held fixed except λ
    n_lambda             : number of λ values in sweep
    lambda_range_factor  : multiplier above/below λ_c for sweep range
    n_mc                 : Monte Carlo trajectories per λ value
    T                    : simulation horizon (minutes)
    seed                 : random seed
    verbose              : print progress

    Returns
    -------
    dict with keys:
        lambda_vals  : 1-D array of λ values tested
        lambda_c     : analytical critical arrival rate
        W_mean       : mean steady-state W* for each λ  (shape: n_lambda)
        W_std        : std of W* across MC runs          (shape: n_lambda)
        W_all        : all W* samples                    (shape: n_lambda, n_mc)
        D_mean       : mean steady-state D*              (shape: n_lambda)
        regime       : list of 'unconstrained'/'congested' per λ
        analytical_W : analytical W* (nan in congested regime)
    """
    lam_c = base_params.critical_arrival_rate()
    lam_min = lam_c / lambda_range_factor
    lam_max = lam_c * lambda_range_factor

    # Log-spaced to get good resolution near the threshold
    lambda_vals = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lambda)

    W_mean = np.zeros(n_lambda)
    W_std  = np.zeros(n_lambda)
    W_all  = np.zeros((n_lambda, n_mc))
    D_mean = np.zeros(n_lambda)
    regimes = []
    analytical_W = np.zeros(n_lambda)

    if verbose:
        print(f"[Threshold] λ_c = {lam_c:.4f}  |  "
              f"sweep: [{lam_min:.4f}, {lam_max:.4f}]  |  {n_lambda} points")

    for i, lam in enumerate(lambda_vals):
        # Build params for this λ
        p = ModelParams(
            lam=lam,
            gamma_I=base_params.gamma_I,
            gamma_D=base_params.gamma_D,
            gamma_P=base_params.gamma_P,
            alpha=base_params.alpha,
            p=base_params.p,
            y=base_params.y,
        )

        # Monte Carlo
        mc = run_monte_carlo(p, n_trajectories=n_mc, T=T, seed=seed + i)

        W_all[i, :] = mc["steady"][:, 1]  # steady W for each trajectory
        W_mean[i]   = W_all[i, :].mean()
        W_std[i]    = W_all[i, :].std()
        D_mean[i]   = mc["steady"][:, 2].mean()

        # Analytical
        ss = analytical_steady_state(p)
        regimes.append(ss["regime"])
        analytical_W[i] = ss["W_star"] if ss["regime"] == "unconstrained" else np.nan

        if verbose:
            regime_str = "CONGEST" if ss["regime"] == "congested" else "stable "
            print(f"  λ={lam:.4f}  {regime_str}  W*={W_mean[i]:.2f}±{W_std[i]:.2f}")

    return {
        "lambda_vals":   lambda_vals,
        "lambda_c":      lam_c,
        "W_mean":        W_mean,
        "W_std":         W_std,
        "W_all":         W_all,
        "D_mean":        D_mean,
        "regime":        regimes,
        "analytical_W":  analytical_W,
    }


def capacity_sweep(
    base_params: ModelParams,
    y_values: np.ndarray = None,
    n_mc: int = 100,
    T: float = 480.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Sweep physician capacity y while holding λ fixed.

    Shows how increasing capacity shifts λ_c, providing direct operational
    insight: how many additional physician slots are needed to move the
    current arrival rate into the unconstrained regime?

    Parameters
    ----------
    base_params : ModelParams (λ fixed at base_params.lam)
    y_values    : array of capacity values to test;
                  defaults to [1, 2, ..., 12]
    n_mc, T, seed, verbose : as in lambda_sweep

    Returns
    -------
    dict with keys:
        y_values    : capacity values tested
        lambda_c    : analytical λ_c for each y value
        W_mean      : mean steady-state W*
        W_std       : std of W*
        regime      : regime at base_params.lam for each y
    """
    if y_values is None:
        y_values = np.arange(1, 13, dtype=float)

    n_y = len(y_values)
    lam_c_arr = np.zeros(n_y)
    W_mean = np.zeros(n_y)
    W_std  = np.zeros(n_y)
    regimes = []

    if verbose:
        print(f"[Capacity sweep] λ_fixed = {base_params.lam:.4f}")

    for i, y in enumerate(y_values):
        p = ModelParams(
            lam=base_params.lam,
            gamma_I=base_params.gamma_I,
            gamma_D=base_params.gamma_D,
            gamma_P=base_params.gamma_P,
            alpha=base_params.alpha,
            p=base_params.p,
            y=y,
        )

        lam_c_arr[i] = p.critical_arrival_rate()

        mc = run_monte_carlo(p, n_trajectories=n_mc, T=T, seed=seed + i)
        W_mean[i] = mc["steady"][:, 1].mean()
        W_std[i]  = mc["steady"][:, 1].std()

        ss = analytical_steady_state(p)
        regimes.append(ss["regime"])

        if verbose:
            regime_str = "CONGEST" if ss["regime"] == "congested" else "stable "
            print(f"  y={y:.1f}  λ_c={lam_c_arr[i]:.4f}  {regime_str}  "
                  f"W*={W_mean[i]:.2f}")

    return {
        "y_values":  y_values,
        "lambda_c":  lam_c_arr,
        "W_mean":    W_mean,
        "W_std":     W_std,
        "regime":    regimes,
    }


def estimate_empirical_threshold(
    sweep_result: dict,
    method: str = "gradient",
) -> float:
    """
    Estimate the empirical congestion threshold from a lambda_sweep result.

    Two methods:
        'gradient' : λ where |dW*/dλ| is maximized (steepest change)
        'regime'   : first λ where MC-mean W* exceeds 2× analytical W*

    Parameters
    ----------
    sweep_result : output of lambda_sweep
    method       : 'gradient' or 'regime'

    Returns
    -------
    lambda_c_empirical : float
    """
    lams = sweep_result["lambda_vals"]
    W    = sweep_result["W_mean"]

    if method == "gradient":
        # Numerical gradient of W* w.r.t. λ — peaks at the threshold
        dW = np.gradient(W, lams)
        idx = np.argmax(np.abs(dW))
        return float(lams[idx])

    elif method == "regime":
        # First λ where simulated W* exceeds 2× analytical prediction
        analytical = sweep_result["analytical_W"]
        for i, (lam, w_sim, w_ana) in enumerate(zip(lams, W, analytical)):
            if not np.isnan(w_ana) and w_sim > 2.0 * w_ana:
                return float(lam)
        return float(lams[-1])  # never exceeded — return max

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'gradient' or 'regime'.")