"""Static publication-quality figures.

For a single (controller, scenario) run, produce a 3-panel figure showing:
    1. Reference vs actual velocity
    2. Tracking error vs time
    3. Control input vs time
A small textbox in the corner summarises the headline KPIs.

Designed for crisp insertion into the .pptx deck — 300-dpi PNG + vector PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..utils.kpi import KPIs
from ..utils.runner import SimulationLog


# Colour palette consistent across all figures. Maps controller name -> colour.
CONTROLLER_COLOURS = {
    "pid":  "#1f77b4",   # blue
    "ipid": "#2ca02c",   # green
    "mpc":  "#d62728",   # red
    "impc": "#9467bd",   # purple
}


def plot_single_run(
    log: SimulationLog,
    kpis: KPIs,
    controller_name: str,
    scenario_name: str,
    scenario_description: str,
    out_path: Path,
    show: bool = False,
) -> Path:
    """Render the 3-panel figure for a single run. Returns the saved PNG path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    colour = CONTROLLER_COLOURS.get(controller_name, "#333333")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    fig.suptitle(
        f"Scenario {scenario_name}  —  {scenario_description}\n"
        f"Controller: {controller_name.upper()}",
        fontsize=12, fontweight="bold",
    )

    # --- Panel 1: velocity tracking -------------------------------------------
    ax = axes[0]
    ax.plot(log.t, log.v_ref, color="#888888", lw=2.0, ls="--", label="Reference")
    ax.plot(log.t, log.v_true, color=colour, lw=1.7, label=f"{controller_name.upper()} (true)")
    ax.fill_between(log.t, log.v_ref, log.v_true, color=colour, alpha=0.10)
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity tracking", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # --- Panel 2: tracking error ---------------------------------------------
    err = log.v_ref - log.v_true
    ax = axes[1]
    ax.plot(log.t, err, color=colour, lw=1.5)
    ax.axhline(0.0, color="#444444", lw=0.6)
    ax.set_ylabel("Error [m/s]")
    ax.set_title(f"Tracking error  (RMSE = {kpis.rmse_velocity:.4f} m/s)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: control input -----------------------------------------------
    ax = axes[2]
    ax.plot(log.t, log.u, color=colour, lw=1.3)
    ax.axhline(0.0, color="#444444", lw=0.6)
    ax.axhline(1.0, color="#bbbbbb", lw=0.5, ls=":")
    ax.axhline(-1.0, color="#bbbbbb", lw=0.5, ls=":")
    ax.set_ylabel("u  (throttle/brake fraction)")
    ax.set_xlabel("Time [s]")
    ax.set_title(
        f"Control effort  (∫u² dt = {kpis.control_effort:.3f}, "
        f"mean compute = {kpis.mean_compute_us:.1f} µs)",
        fontsize=10, loc="left",
    )
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

    # --- KPI textbox ---------------------------------------------------------
    settling = f"{kpis.settling_time:.2f} s" if not np.isnan(kpis.settling_time) else "—"
    text = (
        f"RMSE        : {kpis.rmse_velocity:.4f} m/s\n"
        f"IAE         : {kpis.iae_velocity:.3f} m·s/s\n"
        f"Max |err|   : {kpis.max_abs_error:.4f} m/s\n"
        f"Overshoot   : {kpis.overshoot_pct:.2f} %\n"
        f"Settling    : {settling}\n"
        f"Effort      : {kpis.control_effort:.3f}\n"
        f"Compute     : {kpis.mean_compute_us:.1f} µs/tick\n"
        f"Final s     : {kpis.final_position:.2f} m"
    )
    axes[0].text(
        0.015, 0.97, text,
        transform=axes[0].transAxes, fontsize=8,
        family="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#999999", lw=0.7),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    if show:
        plt.show()
    plt.close(fig)
    return out_path.with_suffix(".png")
