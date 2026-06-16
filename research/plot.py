"""
research/plot.py
----------------
Figure generation for the paper. All functions take pre-computed result
dicts from model.py, sensitivity.py, and threshold.py — no simulation
logic lives here. This keeps figures reproducible from saved results.

Figures produced:
    Figure 1 (fig_convergence)   : MC ensemble convergence to steady state
    Figure 2 (fig_threshold)     : λ sweep showing congestion transition
    Figure 3 (fig_sobol)         : Sobol sensitivity indices bar chart
    Figure 4 (fig_variance_table): variance comparison hybrid vs decoupled
    Figure 5 (fig_capacity_sweep): λ_c vs physician capacity y

Usage:
    from research.plot import fig_convergence, fig_threshold, fig_sobol
    fig_convergence(mc_result, params).savefig("figures/fig1_convergence.pdf")
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Use a clean style suitable for academic publication
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

COMP_NAMES  = ["Intake ($I$)", "Waiting ($W$)", "Physician ($D$)",
               "Pharmacy ($P$)", "Discharged ($X$)"]
COMP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ---------------------------------------------------------------------------
# Figure 1: MC convergence
# ---------------------------------------------------------------------------

def fig_convergence(
    mc_result: dict,
    params,
    compartments: list = None,
    analytical_ss: dict = None,
    n_sample_traces: int = 10,
) -> plt.Figure:
    """
    Time series showing MC ensemble mean ± 1σ with sample trajectories.

    Parameters
    ----------
    mc_result      : output of run_monte_carlo
    params         : ModelParams used to generate mc_result
    compartments   : list of compartment indices to plot (default: [0,1,2])
    analytical_ss  : output of analytical_steady_state (for dashed reference)
    n_sample_traces: number of individual trajectories to show faintly

    Returns
    -------
    matplotlib Figure
    """
    if compartments is None:
        compartments = [0, 1, 2]  # I, W, D

    t   = mc_result["t"]
    mu  = mc_result["mean"]
    sig = mc_result["std"]
    all_traj = mc_result["all"]

    n_panels = len(compartments)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, ci in zip(axes, compartments):
        color = COMP_COLORS[ci]

        # Sample individual trajectories (faint)
        rng = np.random.default_rng(0)
        idx = rng.choice(all_traj.shape[0],
                         size=min(n_sample_traces, all_traj.shape[0]),
                         replace=False)
        for j in idx:
            ax.plot(t / 60, all_traj[j, ci, :], color=color,
                    alpha=0.12, linewidth=0.6)

        # Mean ± 1σ band
        ax.fill_between(t / 60, mu[ci] - sig[ci], mu[ci] + sig[ci],
                        color=color, alpha=0.25, label=r"Mean $\pm 1\sigma$")
        ax.plot(t / 60, mu[ci], color=color, linewidth=2.0, label="MC mean")

        # Analytical steady state
        if analytical_ss is not None and analytical_ss["regime"] == "unconstrained":
            ss_key = ["I_star", "W_star", "D_star", "P_star", None][ci]
            if ss_key:
                ax.axhline(analytical_ss[ss_key], color="black", linestyle="--",
                           linewidth=1.2, label=r"$X^*$ (analytical)")

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Compartment occupancy")
        ax.set_title(COMP_NAMES[ci])
        ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        f"Monte Carlo convergence  "
        f"($M={all_traj.shape[0]}$, $\\lambda={params.lam:.2f}$, $T=480$ min)",
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Congestion threshold sweep
# ---------------------------------------------------------------------------

def fig_threshold(sweep_result: dict) -> plt.Figure:
    """
    Mean steady-state W* vs λ, showing the congestion transition at λ_c.
    """
    lams      = sweep_result["lambda_vals"]
    W_mean    = sweep_result["W_mean"]
    W_std     = sweep_result["W_std"]
    lam_c     = sweep_result["lambda_c"]
    W_ana     = sweep_result["analytical_W"]
    regimes   = sweep_result["regime"]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Shade congested region
    ax.axvspan(lam_c, lams[-1], alpha=0.08, color="red", label="Congested regime")

    # Analytical W* (unconstrained only)
    valid = ~np.isnan(W_ana)
    ax.plot(lams[valid], W_ana[valid], "k--", linewidth=1.5,
            label=r"$W^*$ (analytical)")

    # MC results with error bars
    ax.errorbar(lams, W_mean, yerr=W_std, fmt="o", color=COMP_COLORS[1],
                markersize=4, capsize=3, linewidth=1.2, elinewidth=0.8,
                label=r"$W^*$ (MC mean $\pm 1\sigma$)")

    ax.axvline(lam_c, color="crimson", linestyle="-.", linewidth=1.5,
               label=rf"$\lambda_c = {lam_c:.3f}$")

    ax.set_xlabel(r"Arrival rate $\lambda$ (patients/min)")
    ax.set_ylabel(r"Steady-state waiting occupancy $W^*$")
    ax.set_title("Congestion threshold: simulation vs analytical")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Sobol sensitivity indices
# ---------------------------------------------------------------------------

def fig_sobol(sensitivity_result: dict, output_label: str = r"$W^*$") -> plt.Figure:
    """
    Grouped bar chart of S1 and ST Sobol indices for all parameters.

    Parameters
    ----------
    sensitivity_result : output of run_sensitivity_analysis
    output_label       : label for the output quantity (for title)
    """
    Si     = sensitivity_result["sobol_indices"]
    names  = sensitivity_result["problem"]["names"]
    S1     = Si["S1"]
    ST     = Si["ST"]
    S1_ci  = Si["S1_conf"]
    ST_ci  = Si["ST_conf"]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x - w/2, S1, width=w, color="#4C72B0", alpha=0.85,
                   label=r"$S_i$ (first-order)", yerr=S1_ci,
                   capsize=4, error_kw={"linewidth": 1.0})
    bars2 = ax.bar(x + w/2, ST, width=w, color="#DD8452", alpha=0.85,
                   label=r"$S_i^T$ (total-effect)", yerr=ST_ci,
                   capsize=4, error_kw={"linewidth": 1.0})

    # Annotate top bars
    for bar in bars2:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r"$\lambda$", r"$\gamma_I$", r"$\gamma_D$",
         r"$\gamma_P$", r"$\alpha$", r"$p$", r"$y$"],
        fontsize=11
    )
    ax.set_ylabel("Sobol index")
    ax.set_ylim(0, min(1.0, max(ST) * 1.3))
    ax.set_title(f"Variance-based sensitivity analysis  (output: {output_label})")
    ax.legend(frameon=False)
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Variance comparison table (hybrid vs decoupled)
# ---------------------------------------------------------------------------

def fig_variance_comparison(
    hybrid_mc: dict,
    decoupled_mc: dict,
) -> plt.Figure:
    """
    Bar chart comparing compartment-level steady-state variance between
    the hybrid model and a decoupled stochastic baseline.

    Parameters
    ----------
    hybrid_mc   : run_monte_carlo output for hybrid model
    decoupled_mc: run_monte_carlo output for decoupled model
                  (generate by calling with alpha=0, independent transitions)
    """
    comp_labels = ["Intake\n$I$", "Waiting\n$W$", "Physician\n$D$",
                   "Pharmacy\n$P$"]
    n = len(comp_labels)
    x = np.arange(n)
    w = 0.35

    hybrid_var   = hybrid_mc["steady"][:, :n].var(axis=0)
    decoupled_var = decoupled_mc["steady"][:, :n].var(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x - w/2, hybrid_var,   width=w, color="#55A868", alpha=0.85,
           label="Hybrid PDMP")
    ax.bar(x + w/2, decoupled_var, width=w, color="#C44E52", alpha=0.85,
           label="Decoupled stochastic")

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel(r"Steady-state variance $\mathbb{V}[X^*]$")
    ax.set_title("Variance reduction: hybrid vs.\ decoupled formulation")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Capacity sweep
# ---------------------------------------------------------------------------

def fig_capacity_sweep(capacity_result: dict, base_lam: float) -> plt.Figure:
    """
    Shows how λ_c shifts as physician capacity y increases,
    with the fixed arrival rate λ shown as a horizontal reference.
    """
    y_vals  = capacity_result["y_values"]
    lam_c   = capacity_result["lambda_c"]
    W_mean  = capacity_result["W_mean"]
    W_std   = capacity_result["W_std"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: λ_c vs y
    ax1.plot(y_vals, lam_c, "o-", color="#4C72B0", linewidth=2, markersize=5,
             label=r"$\lambda_c(y)$")
    ax1.axhline(base_lam, color="crimson", linestyle="--", linewidth=1.5,
                label=rf"Current $\lambda = {base_lam:.3f}$")

    # Shade where λ < λ_c (stable)
    stable = lam_c > base_lam
    if stable.any():
        y_min_stable = y_vals[stable][0]
        ax1.axvspan(y_min_stable, y_vals[-1], alpha=0.1, color="green",
                    label="Stable regime")

    ax1.set_xlabel("Physician capacity $y$")
    ax1.set_ylabel(r"$\lambda_c$")
    ax1.set_title(r"Congestion threshold vs.\ capacity")
    ax1.legend(frameon=False)

    # Right: W* vs y
    ax2.errorbar(y_vals, W_mean, yerr=W_std, fmt="s-", color="#DD8452",
                 markersize=5, capsize=3, linewidth=1.8, elinewidth=0.8)
    ax2.set_xlabel("Physician capacity $y$")
    ax2.set_ylabel(r"Steady-state $W^*$")
    ax2.set_title("Waiting occupancy vs.\ capacity")

    fig.suptitle("Physician capacity sensitivity", fontsize=13)
    fig.tight_layout()
    return fig