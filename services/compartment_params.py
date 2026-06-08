"""
compartment_params.py
─────────────────────
Extracts all model parameters from the ScanLog sheet that's already
being written by scan_service.py.

Expected ScanLog columns:
    patient_uuid | timestamp | action | source | metadata

Relevant action values:
    MOVE_COMPARTMENT  →  metadata contains the *destination* compartment
    SCAN_LOOKUP       →  patient arrival proxy (first touch = intake)

Returns a ModelParams dataclass consumed by the ODE solver.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


# ── Compartment constants (must match config.py COMPARTMENTS) ─────────────
WAITING  = "WAITING"
DOCTOR   = "DOCTOR"
PHARMACY = "PHARMACY"
DISCHARGE = "DISCHARGE"

TRACKED_COMPARTMENTS = [WAITING, DOCTOR, PHARMACY]


@dataclass
class ModelParams:
    """All parameters needed to define and solve the ODE system."""

    # Arrival rate vector — one value per hour bin (length 24)
    # λ(t) is reconstructed as a step function over the day
    lambda_hourly: List[float] = field(default_factory=lambda: [0.0] * 24)

    # Transition rates (units: patients / hour)
    alpha: float = 1.0   # Waiting  → Doctor
    beta:  float = 1.0   # Doctor   → Pharmacy   (= p * mu_D)
    gamma: float = 1.0   # Doctor   → Discharge  (= (1-p) * mu_D)
    delta: float = 1.0   # Pharmacy → Discharge

    # LWBS (Left Without Being Seen) leak from Waiting
    # Set to 0.0 to disable this term entirely
    mu_lwbs: float = 0.0

    # Branching probability: P(Doctor → Pharmacy)
    p_pharmacy: float = 0.5

    # Sample sizes used to estimate each rate (for confidence display)
    n_waiting_transitions: int = 0
    n_doctor_transitions:  int = 0
    n_pharmacy_transitions: int = 0
    n_arrivals:            int = 0

    # Mean dwell times in minutes (human-readable, not used in solver)
    mean_wait_min:     float = 0.0
    mean_doctor_min:   float = 0.0
    mean_pharmacy_min: float = 0.0


def extract_params(log_df: pd.DataFrame, lwbs_enabled: bool = True) -> ModelParams:
    """
    Main entry point. Pass in the ScanLog DataFrame directly from session state.

    Parameters
    ----------
    log_df       : pd.DataFrame  — the ScanLog sheet
    lwbs_enabled : bool          — whether to estimate the LWBS leak rate

    Returns
    -------
    ModelParams instance with all rates populated.
    """
    params = ModelParams()

    if log_df is None or log_df.empty:
        return params  # return defaults if no data yet

    # Normalise column names defensively
    log_df = log_df.copy()
    log_df.columns = [c.strip().lower() for c in log_df.columns]
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce")
    log_df = log_df.dropna(subset=["timestamp"])

    # ── 1. Arrival rate λ(t) ─────────────────────────────────────────────
    # Proxy: first MOVE_COMPARTMENT to WAITING per patient per day
    # (or SCAN_LOOKUP if you prefer to use that as the arrival signal)
    arrivals = _extract_arrivals(log_df)
    params.lambda_hourly, params.n_arrivals = _build_hourly_lambda(arrivals)

    # ── 2. Filter to movement events only ────────────────────────────────
    moves = log_df[log_df["action"] == "MOVE_COMPARTMENT"].copy()
    moves["metadata"] = moves["metadata"].astype(str).str.upper().str.strip()

    # Reconstruct source compartment per patient via previous destination
    moves = moves.sort_values(["patient_uuid", "timestamp"])
    moves["prev_dest"] = moves.groupby("patient_uuid")["metadata"].shift(1)

    # ── 3. Dwell times per compartment ───────────────────────────────────
    moves["dwell_hours"] = (
        moves["timestamp"] - moves.groupby("patient_uuid")["timestamp"].shift(1)
    ).dt.total_seconds() / 3600

    # Only keep rows where dwell is plausible (< 24 h, > 0)
    moves = moves[(moves["dwell_hours"] > 0) & (moves["dwell_hours"] < 24)]

    # Waiting dwell = time until patient moves OUT of Waiting
    w_rows = moves[moves["prev_dest"] == WAITING]
    if not w_rows.empty:
        mean_w = w_rows["dwell_hours"].mean()
        params.alpha            = 1.0 / mean_w if mean_w > 0 else 1.0
        params.mean_wait_min    = mean_w * 60
        params.n_waiting_transitions = len(w_rows)

        if lwbs_enabled:
            lwbs_rows = w_rows[w_rows["metadata"] == DISCHARGE]
            total_w   = len(w_rows)
            if total_w > 0:
                lwbs_frac     = len(lwbs_rows) / total_w
                params.mu_lwbs = lwbs_frac * params.alpha

    # Doctor dwell
    d_rows = moves[moves["prev_dest"] == DOCTOR]
    if not d_rows.empty:
        mean_d = d_rows["dwell_hours"].mean()
        mu_d   = 1.0 / mean_d if mean_d > 0 else 1.0
        params.mean_doctor_min = mean_d * 60
        params.n_doctor_transitions = len(d_rows)

        # Branching probability
        to_pharmacy  = (d_rows["metadata"] == PHARMACY).sum()
        to_discharge = (d_rows["metadata"] == DISCHARGE).sum()
        total_d      = to_pharmacy + to_discharge
        if total_d > 0:
            params.p_pharmacy = to_pharmacy / total_d
        params.beta  = params.p_pharmacy       * mu_d
        params.gamma = (1 - params.p_pharmacy) * mu_d

    # Pharmacy dwell
    p_rows = moves[moves["prev_dest"] == PHARMACY]
    if not p_rows.empty:
        mean_p = p_rows["dwell_hours"].mean()
        params.delta              = 1.0 / mean_p if mean_p > 0 else 1.0
        params.mean_pharmacy_min  = mean_p * 60
        params.n_pharmacy_transitions = len(p_rows)

    return params


# ── Internal helpers ──────────────────────────────────────────────────────

def _extract_arrivals(log_df: pd.DataFrame) -> pd.Series:
    """Return Series of arrival timestamps (first WAITING move per patient per day)."""
    waiting_moves = log_df[
        (log_df["action"] == "MOVE_COMPARTMENT") &
        (log_df["metadata"].astype(str).str.upper().str.strip() == WAITING)
    ].copy()

    if waiting_moves.empty:
        # Fall back to SCAN_LOOKUP events as proxy
        waiting_moves = log_df[log_df["action"] == "SCAN_LOOKUP"].copy()

    if waiting_moves.empty:
        return pd.Series([], dtype="datetime64[ns]")

    # One arrival per patient per calendar day
    waiting_moves["date"] = waiting_moves["timestamp"].dt.date
    first_per_day = waiting_moves.groupby(["patient_uuid", "date"])["timestamp"].min()
    return first_per_day


def _build_hourly_lambda(arrivals: pd.Series) -> tuple:
    """
    Bin arrivals by hour-of-day and average across all observed days.
    Returns (lambda_hourly: List[float], n_total: int).
    """
    if arrivals.empty:
        return [0.0] * 24, 0

    ts = pd.Series(arrivals.values)
    hours = ts.dt.hour

    # Count arrivals per hour bin per day, then average
    df = pd.DataFrame({"hour": hours, "date": ts.dt.date})
    n_days = df["date"].nunique()
    n_days = max(n_days, 1)

    hourly_totals = df.groupby("hour").size()
    lambda_h = [hourly_totals.get(h, 0) / n_days for h in range(24)]

    return lambda_h, int(len(ts))
