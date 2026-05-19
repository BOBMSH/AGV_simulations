"""KPI computation utilities.

Given the per-tick logs from a simulation run, compute a standard set of
performance metrics used across all controller/scenario comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


@dataclass
class KPIs:
    """Standard performance metrics for one (controller, scenario) run."""

    rmse_velocity: float
    iae_velocity: float
    max_abs_error: float
    overshoot_pct: float       # % over the cruise reference
    control_effort: float       # integral of u^2 dt
    settling_time: float        # to ±2% of cruise reference, NaN if never settles
    mean_compute_us: float      # mean per-tick controller compute time
    final_position: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_kpis(
    t: np.ndarray,
    v_ref: np.ndarray,
    v_actual: np.ndarray,
    s_actual: np.ndarray,
    u: np.ndarray,
    compute_times_us: np.ndarray,
    settle_tol: float = 0.02,
) -> KPIs:
    """Compute KPIs from per-tick logs.

    Settling time is measured against the cruise phase: the delay between the
    start of the contiguous cruise window (samples where v_ref equals its
    peak) and the first index after which |error| stays within tol*v_peak for
    the remainder of that window. Returns NaN if the controller never settles
    inside cruise.
    """
    error = v_ref - v_actual
    dt = np.diff(t, prepend=t[0])

    rmse = float(np.sqrt(np.mean(error ** 2)))
    iae = float(np.sum(np.abs(error) * dt))
    max_abs_err = float(np.max(np.abs(error)))

    v_ref_peak = float(np.max(v_ref)) if np.max(v_ref) > 1e-6 else 1.0
    v_peak = float(np.max(v_actual))
    overshoot = max(0.0, 100.0 * (v_peak - v_ref_peak) / v_ref_peak)

    effort = float(np.sum(u * u * dt))

    settle = float("nan")
    tol = settle_tol * v_ref_peak
    at_peak = np.isclose(v_ref, v_ref_peak, rtol=0.0, atol=1e-9)
    if at_peak.any():
        idx_start = int(np.argmax(at_peak))
        end_run = idx_start
        while end_run < len(at_peak) and at_peak[end_run]:
            end_run += 1
        if end_run > idx_start:
            err_window = np.abs(error[idx_start:end_run])
            inside = err_window <= tol
            # cum_inside_from_end[j] = inside[j:].all()
            cum = np.flip(np.minimum.accumulate(np.flip(inside.astype(int))))
            if cum.any():
                first_settled = int(np.argmax(cum))
                settle = float(t[idx_start + first_settled] - t[idx_start])

    mean_compute = float(np.mean(compute_times_us))
    final_s = float(s_actual[-1])

    return KPIs(
        rmse_velocity=rmse,
        iae_velocity=iae,
        max_abs_error=max_abs_err,
        overshoot_pct=overshoot,
        control_effort=effort,
        settling_time=settle,
        mean_compute_us=mean_compute,
        final_position=final_s,
    )
