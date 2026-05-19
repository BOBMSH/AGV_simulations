"""Multi-controller comparison visualisation.

Plots all controllers on one scenario overlaid in a single figure for
direct visual comparison plus a tabular KPI summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..utils.kpi import KPIs
from ..utils.runner import SimulationLog
from .static import CONTROLLER_COLOURS


def plot_comparison(
    runs: List[Tuple[str, SimulationLog, KPIs]],
    scenario_name: str,
    scenario_description: str,
    out_path: Path,
    v_max_fn=None,
    s_target: Optional[float] = None,
    s_tol: float = 0.05,
    show: bool = False,
) -> Path:
    """Render a 4-panel comparison figure.

    Layout:
        (top)    velocity vs time, all controllers + v_ref + v_max(s) envelope
        (mid 1)  tracking error vs time
        (mid 2)  control input vs time
        (bottom) position vs time (with dock target marker, if any)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    title = f"Scenario {scenario_name}  -  {scenario_description}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Panel 1: velocity tracking ------------------------------------------------
    ax = axes[0]
    ref_drawn = False
    for name, log, _ in runs:
        if not ref_drawn:
            ax.plot(log.t, log.v_ref, color="#666666", lw=1.6, ls="--", label="v_ref(t)")
            if v_max_fn is not None:
                vmax = np.array([v_max_fn(si) for si in log.s])
                ax.plot(log.t, vmax, color="#c54a4a", lw=1.0, ls=":", label="v_max(s)")
            ref_drawn = True
        col = CONTROLLER_COLOURS.get(name, "#333")
        ax.plot(log.t, log.v_true, color=col, lw=1.5, label=name.upper())
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity tracking", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # Panel 2: tracking error ---------------------------------------------------
    ax = axes[1]
    for name, log, _ in runs:
        col = CONTROLLER_COLOURS.get(name, "#333")
        ax.plot(log.t, log.v_ref - log.v_true, color=col, lw=1.3, label=name.upper())
    ax.axhline(0.0, color="#444", lw=0.5)
    ax.set_ylabel("v_ref - v_true [m/s]")
    ax.set_title("Tracking error", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # Panel 3: control input ----------------------------------------------------
    ax = axes[2]
    for name, log, _ in runs:
        col = CONTROLLER_COLOURS.get(name, "#333")
        ax.plot(log.t, log.u, color=col, lw=1.1, label=name.upper())
    ax.axhline(0.0, color="#444", lw=0.5)
    ax.axhline(1.0, color="#bbb", lw=0.5, ls=":")
    ax.axhline(-1.0, color="#bbb", lw=0.5, ls=":")
    ax.set_ylabel("u (throttle/brake)")
    ax.set_title("Control input", fontsize=10, loc="left")
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3)

    # Panel 4: position ---------------------------------------------------------
    ax = axes[3]
    for name, log, _ in runs:
        col = CONTROLLER_COLOURS.get(name, "#333")
        ax.plot(log.t, log.s, color=col, lw=1.5, label=name.upper())
    if s_target is not None:
        ax.axhline(s_target, color="#666", lw=0.8, ls="--", label=f"s_target={s_target} m")
        ax.axhspan(s_target - s_tol, s_target + s_tol, color="#888", alpha=0.15,
                   label=f"+/-{s_tol*100:.0f} cm tolerance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # KPI annotation box --------------------------------------------------------
    lines = [f"{'Controller':<8}  {'RMSE_v':>8}  {'IAE':>7}  {'Overshoot':>9}  {'Effort':>7}  {'Compute':>9}  {'Final s':>8}"]
    for name, log, k in runs:
        compute_disp = (f"{k.mean_compute_us:.1f}us"
                        if k.mean_compute_us < 1000.0
                        else f"{k.mean_compute_us/1000.0:.2f}ms")
        lines.append(
            f"{name.upper():<8}  {k.rmse_velocity:>8.4f}  {k.iae_velocity:>7.3f}  "
            f"{k.overshoot_pct:>8.2f}%  {k.control_effort:>7.3f}  {compute_disp:>9}  "
            f"{k.final_position:>8.3f}"
        )
    text = "\n".join(lines)
    fig.text(0.99, 0.005, text, fontsize=8, family="monospace",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#999", lw=0.7))

    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    if show:
        plt.show()
    plt.close(fig)
    return out_path.with_suffix(".png")
