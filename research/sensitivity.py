"""
research/sensitivity.py
-----------------------
Latin Hypercube Sampling (LHS) and Sobol variance-based sensitivity analysis.

Connects directly to the RUSIS space-filling design work:
  - LHS generates a space-filling sample of the parameter space (same
    methodological family as the SLHD/MaxPro designs from the REU)
  - Sobol indices decompose output variance by parameter contribution
  - The Saltelli estimator requires M*(k+2) model runs for k parameters

Workflow (without empirical data):
    1. Define parameter bounds using ModelParams defaults ± uncertainty
    2. Generate LHS sample via SALib
    3. For each sample, run the PDMP and extract steady-state outputs
    4. Estimate Sobol indices from the output ensemble

When empirical data becomes available:
    - Replace `default_bounds()` with `bounds_from_data(estimates, cov)`
    - Everything downstream works identically
"""

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from typing import Callable, Optional
from research.model import ModelParams, run_monte_carlo, analytical_steady_state


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

def default_bounds(params: Optional[ModelParams] = None) -> dict:
    """
    Define the SALib problem dict with parameter bounds.

    Bounds represent ±40% around default values — conservative uncertainty
    for a synthetic experiment. Replace with empirically derived confidence
    intervals when data is available.

    Parameters
    ----------
    params : optional ModelParams to center bounds around;
             uses ModelParams() defaults if None

    Returns
    -------
    SALib problem dict
    """
    if params is None:
        params = ModelParams()

    def pm(val, frac=0.4):
        """Return [val*(1-frac), val*(1+frac)]."""
        return [val * (1 - frac), val * (1 + frac)]

    return {
        "num_vars": 7,
        "names": ModelParams.param_names(),
        "bounds": [
            pm(params.lam),      # arrival rate
            pm(params.gamma_I),  # intake rate
            pm(params.gamma_D),  # doctor rate
            pm(params.gamma_P),  # pharmacy rate
            pm(params.alpha),    # waiting→doctor rate
            [0.1, 0.8],          # routing prob p (bounded to (0,1))
            [2.0, 10.0],         # physician capacity y (discrete operational)
        ],
    }


def bounds_from_estimates(
    estimates: dict,
    relative_uncertainty: float = 0.3,
) -> dict:
    """
    Build SALib problem dict from empirically estimated parameters.

    Call this once you have real data — pass in the MLE estimates
    and this replaces default_bounds().

    Parameters
    ----------
    estimates : dict mapping param names to point estimates
                e.g. {"lambda": 0.6, "gamma_I": 0.18, ...}
    relative_uncertainty : fraction ± around each estimate

    Returns
    -------
    SALib problem dict
    """
    names = ModelParams.param_names()
    bounds = []
    for name in names:
        val = estimates[name]
        lo = val * (1 - relative_uncertainty)
        hi = val * (1 + relative_uncertainty)
        # Enforce positivity and (0,1) for p
        if name == "p":
            lo = max(lo, 0.05)
            hi = min(hi, 0.95)
        bounds.append([max(lo, 1e-6), hi])

    return {"num_vars": 7, "names": names, "bounds": bounds}


# ---------------------------------------------------------------------------
# LHS sampling (space-filling design)
# ---------------------------------------------------------------------------

def lhs_sample(
    problem: dict,
    M: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a Latin Hypercube Sample of the parameter space.

    Uses SALib's Saltelli sampler, which extends LHS with additional
    samples required for Sobol index estimation. Total samples = M*(k+2)
    where k = number of parameters.

    This is the same space-filling design philosophy as the RUSIS work —
    LHS ensures each parameter's marginal distribution is well-covered
    with far fewer runs than a full factorial grid.

    Parameters
    ----------
    problem : SALib problem dict (from default_bounds or bounds_from_estimates)
    M       : base sample size; total runs = M * (k+2)
    seed    : random seed

    Returns
    -------
    param_values : np.ndarray of shape (M*(k+2), k)
    """
    param_values = saltelli.sample(problem, M, calc_second_order=False,
                                   seed=seed)
    print(f"[LHS] Generated {param_values.shape[0]} parameter samples "
          f"({M} base × {problem['num_vars']+2} = {M*(problem['num_vars']+2)})")
    return param_values


# ---------------------------------------------------------------------------
# Model evaluation over LHS sample
# ---------------------------------------------------------------------------

def evaluate_model(
    param_values: np.ndarray,
    output_fn: Optional[Callable] = None,
    T: float = 480.0,
    n_mc: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Evaluate the model at each LHS parameter sample.

    For each row in param_values, constructs a ModelParams, runs
    n_mc Monte Carlo trajectories, and extracts steady-state waiting
    occupancy W* as the primary output (most sensitive compartment per
    the congestion threshold analysis).

    Parameters
    ----------
    param_values : output of lhs_sample, shape (N, 7)
    output_fn    : optional callable(mc_result) -> float
                   Defaults to mean steady-state W (waiting compartment)
                   Swap this to change the output quantity of interest.
    T            : simulation horizon
    n_mc         : Monte Carlo trajectories per parameter sample
                   (keep low here since we have many samples; increase for
                   final paper figures)
    seed         : base random seed
    verbose      : print progress every 50 samples

    Returns
    -------
    Y : np.ndarray of shape (N,) — one scalar output per parameter sample
    """
    if output_fn is None:
        # Default: mean steady-state waiting occupancy across MC runs
        def output_fn(mc_result):
            return mc_result["steady"][:, 1].mean()  # col 1 = W

    N = param_values.shape[0]
    Y = np.zeros(N)

    for i, row in enumerate(param_values):
        params = ModelParams.from_array(row)

        # Clamp p to (0,1) in case sampler drifts slightly
        params.p = float(np.clip(params.p, 1e-3, 1 - 1e-3))
        params.y = max(params.y, 1.0)

        mc = run_monte_carlo(params, n_trajectories=n_mc, T=T,
                             seed=seed + i)
        Y[i] = output_fn(mc)

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] λ={params.lam:.3f}  W*={Y[i]:.2f}")

    return Y


def evaluate_model_deterministic(
    param_values: np.ndarray,
    output_fn: Optional[Callable] = None,
) -> np.ndarray:
    """
    Fast version: evaluate analytical steady state instead of running MC.

    Use this for initial exploration and debugging — instant results.
    Switch to evaluate_model() for the final paper figures.

    Falls back to 0.0 for congested-regime samples (W* diverges there).
    """
    if output_fn is None:
        def output_fn(ss): return ss["W_star"]

    N = param_values.shape[0]
    Y = np.zeros(N)

    for i, row in enumerate(param_values):
        params = ModelParams.from_array(row)
        params.p = float(np.clip(params.p, 1e-3, 1 - 1e-3))
        params.y = max(params.y, 1.0)

        ss = analytical_steady_state(params)
        if ss["regime"] == "unconstrained":
            Y[i] = output_fn(ss)
        else:
            Y[i] = np.nan  # congested — W* diverges, handled separately

    return Y


# ---------------------------------------------------------------------------
# Sobol index estimation
# ---------------------------------------------------------------------------

def compute_sobol_indices(
    problem: dict,
    Y: np.ndarray,
    print_summary: bool = True,
) -> dict:
    """
    Estimate first-order (S1) and total-effect (ST) Sobol indices.

    S1_i = Var_{θ_i}[E[Y | θ_i]] / Var[Y]
        — fraction of output variance explained by θ_i alone

    ST_i = 1 - Var_{θ_{-i}}[E[Y | θ_{-i}]] / Var[Y]
        — total contribution of θ_i including all interactions

    Parameters
    ----------
    problem : SALib problem dict used to generate param_values
    Y       : model output array from evaluate_model, shape (N,)
    print_summary : whether to print a ranked table of indices

    Returns
    -------
    Si : SALib analysis dict with keys 'S1', 'ST', 'S1_conf', 'ST_conf'
    """
    # Drop NaN entries (congested regime samples)
    if np.any(np.isnan(Y)):
        n_nan = np.isnan(Y).sum()
        print(f"[Sobol] Warning: {n_nan} NaN outputs (congested regime) "
              f"replaced with max finite value for index estimation.")
        Y = Y.copy()
        Y[np.isnan(Y)] = np.nanmax(Y)

    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    if print_summary:
        print("\n── Sobol Sensitivity Indices ──────────────────────")
        print(f"{'Parameter':<12} {'S1':>8} {'S1_conf':>10} {'ST':>8} {'ST_conf':>10}")
        print("─" * 52)
        names = problem["names"]
        for i, name in enumerate(names):
            print(f"{name:<12} {Si['S1'][i]:>8.4f} {Si['S1_conf'][i]:>10.4f} "
                  f"{Si['ST'][i]:>8.4f} {Si['ST_conf'][i]:>10.4f}")
        print(f"\nTop driver: {names[np.argmax(Si['ST'])]}")
        print("────────────────────────────────────────────────────\n")

    return Si


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    params: Optional[ModelParams] = None,
    M: int = 128,
    T: float = 480.0,
    n_mc: int = 30,
    fast: bool = True,
    seed: int = 42,
) -> dict:
    """
    End-to-end sensitivity analysis pipeline.

    Parameters
    ----------
    params  : center point for bounds; uses defaults if None
    M       : base LHS sample size (total = M*(k+2) = M*9 for k=7)
    T       : simulation horizon
    n_mc    : MC trajectories per sample (ignored if fast=True)
    fast    : if True, uses analytical steady state (instant);
              if False, runs full Monte Carlo (accurate, slow)
    seed    : random seed

    Returns
    -------
    dict with keys: problem, param_values, Y, sobol_indices
    """
    problem = default_bounds(params)
    param_values = lhs_sample(problem, M=M, seed=seed)

    print(f"[Sensitivity] Running {'analytical' if fast else 'MC'} evaluation "
          f"on {param_values.shape[0]} samples...")

    if fast:
        Y = evaluate_model_deterministic(param_values)
    else:
        Y = evaluate_model(param_values, T=T, n_mc=n_mc, seed=seed)

    Si = compute_sobol_indices(problem, Y)

    return {
        "problem":       problem,
        "param_values":  param_values,
        "Y":             Y,
        "sobol_indices": Si,
    }