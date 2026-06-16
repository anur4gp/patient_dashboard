"""
research/run_experiments.py
---------------------------
Entry point for generating all paper figures and results.

Run from the patient_dashboard root directory:
    python -m research.run_experiments

Outputs written to:  research/output/
    figures/fig1_convergence.pdf
    figures/fig2_threshold.pdf
    figures/fig3_sobol.pdf
    figures/fig5_capacity.pdf
    results/mc_result.npz
    results/sensitivity_result.npz
    results/threshold_result.npz

Designed to run in two modes:
    FAST=True  : uses analytical steady state for sensitivity (seconds)
    FAST=False : full Monte Carlo everywhere (minutes, for final figures)

Set FAST=False for paper submission runs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from research.model       import ModelParams, run_monte_carlo, analytical_steady_state
from research.sensitivity import run_sensitivity_analysis
from research.threshold   import lambda_sweep, capacity_sweep, estimate_empirical_threshold
from research.plot        import (fig_convergence, fig_threshold,
                                  fig_sobol, fig_capacity_sweep)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAST        = True    # flip to False for final paper runs
SEED        = 42
T           = 480.0   # 8-hour clinic day (minutes)
N_MC        = 200     # MC trajectories for convergence figure
N_MC_SWEEP  = 80      # MC per λ value in threshold sweep
SOBOL_M     = 128     # base LHS size; total = 128*9 = 1152 evaluations

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RES_DIR = os.path.join(OUT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Default synthetic parameters (replace with empirical estimates when available)
params = ModelParams(
    lam=0.5,
    gamma_I=0.20,
    gamma_D=0.05,
    gamma_P=0.10,
    alpha=0.15,
    p=0.40,
    y=5.0,
)


# ---------------------------------------------------------------------------
# 1. Convergence (Figure 1)
# ---------------------------------------------------------------------------

def run_convergence():
    print("\n══ [1/4] Convergence analysis ══")
    mc = run_monte_carlo(params, n_trajectories=N_MC, T=T, seed=SEED)
    ss = analytical_steady_state(params)

    print(f"  λ_c = {params.critical_arrival_rate():.4f}  "
          f"(current λ = {params.lam:.4f} → {ss['regime']})")
    print(f"  Analytical W* = {ss['W_star']:.3f}  |  "
          f"MC W* = {mc['steady'][:,1].mean():.3f} ± {mc['steady'][:,1].std():.3f}")

    fig = fig_convergence(mc, params, compartments=[0, 1, 2], analytical_ss=ss)
    path = os.path.join(FIG_DIR, "fig1_convergence.pdf")
    fig.savefig(path)
    print(f"  → saved {path}")
    plt.close(fig)

    np.savez(os.path.join(RES_DIR, "mc_result.npz"),
             t=mc["t"], mean=mc["mean"], std=mc["std"], steady=mc["steady"])


# ---------------------------------------------------------------------------
# 2. Threshold sweep (Figure 2)
# ---------------------------------------------------------------------------

def run_threshold():
    print("\n══ [2/4] Threshold sweep ══")
    sweep = lambda_sweep(
        base_params=params,
        n_lambda=30,
        lambda_range_factor=2.0,
        n_mc=N_MC_SWEEP,
        T=T,
        seed=SEED,
    )

    lam_c_empirical = estimate_empirical_threshold(sweep, method="gradient")
    print(f"  Analytical λ_c = {sweep['lambda_c']:.4f}")
    print(f"  Empirical  λ_c = {lam_c_empirical:.4f}  "
          f"(error: {abs(lam_c_empirical - sweep['lambda_c'])/sweep['lambda_c']*100:.1f}%)")

    fig = fig_threshold(sweep)
    path = os.path.join(FIG_DIR, "fig2_threshold.pdf")
    fig.savefig(path)
    print(f"  → saved {path}")
    plt.close(fig)

    np.savez(os.path.join(RES_DIR, "threshold_result.npz"),
             lambda_vals=sweep["lambda_vals"],
             W_mean=sweep["W_mean"], W_std=sweep["W_std"],
             lambda_c=np.array([sweep["lambda_c"]]),
             analytical_W=sweep["analytical_W"])


# ---------------------------------------------------------------------------
# 3. Sensitivity analysis (Figure 3)
# ---------------------------------------------------------------------------

def run_sensitivity():
    print(f"\n══ [3/4] Sensitivity analysis (fast={FAST}) ══")
    result = run_sensitivity_analysis(
        params=params,
        M=SOBOL_M,
        T=T,
        n_mc=30,
        fast=FAST,
        seed=SEED,
    )

    fig = fig_sobol(result)
    path = os.path.join(FIG_DIR, "fig3_sobol.pdf")
    fig.savefig(path)
    print(f"  → saved {path}")
    plt.close(fig)

    Si = result["sobol_indices"]
    np.savez(os.path.join(RES_DIR, "sensitivity_result.npz"),
             S1=Si["S1"], ST=Si["ST"],
             S1_conf=Si["S1_conf"], ST_conf=Si["ST_conf"],
             Y=result["Y"])


# ---------------------------------------------------------------------------
# 4. Capacity sweep (Figure 5)
# ---------------------------------------------------------------------------

def run_capacity():
    print("\n══ [4/4] Capacity sweep ══")
    cap = capacity_sweep(
        base_params=params,
        y_values=np.arange(1, 13, dtype=float),
        n_mc=N_MC_SWEEP,
        T=T,
        seed=SEED,
    )

    fig = fig_capacity_sweep(cap, base_lam=params.lam)
    path = os.path.join(FIG_DIR, "fig5_capacity.pdf")
    fig.savefig(path)
    print(f"  → saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"patient_dashboard / research experiments")
    print(f"  params: {params}")
    print(f"  λ_c = {params.critical_arrival_rate():.4f}")
    print(f"  mode: {'FAST (analytical)' if FAST else 'FULL Monte Carlo'}")

    run_convergence()
    run_threshold()
    run_sensitivity()
    run_capacity()

    print("\n✓ All experiments complete. Figures in research/output/figures/")