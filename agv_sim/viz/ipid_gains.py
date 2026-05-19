"""IPID-specific figure: tracking + live gain evolution.

Shows how the NN gain scheduler adapts Kp, Ki, Kd to the AGV's current
operating point (surface Cr, payload). The bottom panel breaks out the
three gain curves, with vertical guides marking surface / payload zone
boundaries when the run is on Scenario B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..utils.kpi import KPIs
from ..utils.runner import SimulationLog
from .static import CONTROLLER_COLOURS


def plot_ipid_run(
    log: SimulationLog,
    kpis: KPIs,
    scenario_name: str,
    scenario_description: str,
    out_path: Path,
    show: bool = False,
) -> Path:
    """4-panel figure for an IPID run: velocity, error, control, live gains."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    col = CONTROLLER_COLOURS.get("ipid", "#2ca02c")

    # Extract per-tick gains from the diagnostics list.
    kp = np.array([d.get("kp", np.nan) for d in log.diagnostics])
    ki = np.array([d.get("ki", np.nan) for d in log.diagnostics])
    kd = np.array([d.get("kd", np.nan) for d in log.diagnostics])
    kp_nn = np.array([d.get("kp_nn", np.nan) for d in log.diagnostics])
    ki_nn = np.array([d.get("ki_nn", np.nan) for d in log.diagnostics])
    kd_nn = np.array([d.get("kd_nn", np.nan) for d in log.diagnostics])

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1.2, 1, 1, 1.6]})
    fig.suptitle(
        f"Scenario {scenario_name}  -  {scenario_description}\n"
        f"Controller: IPID  (NN-driven gain scheduling)",
        fontsize=12, fontweight="bold",
    )

    # --- Optional zone bands for Scenario B ---
    zone_bands = []
    if scenario_name == "B":
        from ..scenarios.scenario_b_friction import SURFACES
        # Convert path-position zones to time approximate via integrated v_ref.
        dt = log.t[1] - log.t[0] if len(log.t) > 1 else 0.05
        cum_s = np.cumsum(log.v_true) * dt
        for seg in SURFACES:
            # Find first time when AGV crosses seg.s_start and seg.s_end.
            idx_start = int(np.argmax(cum_s >= seg.s_start)) if (cum_s >= seg.s_start).any() else 0
            idx_end = int(np.argmax(cum_s >= seg.s_end)) if (cum_s >= seg.s_end).any() else len(log.t) - 1
            zone_bands.append((log.t[idx_start], log.t[idx_end], seg.label))

    def shade_zones(ax):
        if not zone_bands:
            return
        cmap = ["#fcfcfc", "#f3f3f8", "#e8eef8", "#fff2e0"]
        for i, (t0, t1, lbl) in enumerate(zone_bands):
            ax.axvspan(t0, t1, color=cmap[i % len(cmap)], alpha=0.6,
                       zorder=-10)

    # --- Panel 1: velocity tracking ---
    ax = axes[0]
    shade_zones(ax)
    ax.plot(log.t, log.v_ref, color="#666", lw=1.6, ls="--", label="v_ref")
    ax.plot(log.t, log.v_true, color=col, lw=1.6, label="v_true (IPID)")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity tracking", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # --- Panel 2: tracking error ---
    ax = axes[1]
    shade_zones(ax)
    err = log.v_ref - log.v_true
    ax.plot(log.t, err, color=col, lw=1.3)
    ax.axhline(0.0, color="#444", lw=0.6)
    ax.set_ylabel("Error [m/s]")
    ax.set_title(f"Tracking error  (RMSE = {kpis.rmse_velocity:.4f} m/s)",
                 fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: control input ---
    ax = axes[2]
    shade_zones(ax)
    ax.plot(log.t, log.u, color=col, lw=1.1)
    ax.axhline(0.0, color="#444", lw=0.5)
    ax.axhline(1.0, color="#bbb", lw=0.5, ls=":")
    ax.axhline(-1.0, color="#bbb", lw=0.5, ls=":")
    ax.set_ylabel("u  (throttle/brake)")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Control input", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: live gains from the NN ---
    ax = axes[3]
    shade_zones(ax)
    # Plot smoothed (applied) gains on a left axis, raw NN output as faint dashed.
    line_kp, = ax.plot(log.t, kp, color="#1f77b4", lw=1.7, label="Kp (applied)")
    line_ki, = ax.plot(log.t, ki, color="#d62728", lw=1.7, label="Ki (applied)")
    ax.plot(log.t, kp_nn, color="#1f77b4", lw=0.8, ls=":", alpha=0.7,
            label="Kp raw NN")
    ax.plot(log.t, ki_nn, color="#d62728", lw=0.8, ls=":", alpha=0.7,
            label="Ki raw NN")
    ax.set_ylabel("Kp, Ki", color="#000")
    ax.tick_params(axis="y", labelcolor="#000")
    ax.set_xlabel("Time [s]")
    ax.set_title("NN-driven gain evolution",
                 fontsize=10, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Kd on a twin axis (smaller numerical range).
    ax2 = ax.twinx()
    line_kd, = ax2.plot(log.t, kd, color="#2ca02c", lw=1.7, label="Kd (applied)")
    ax2.plot(log.t, kd_nn, color="#2ca02c", lw=0.8, ls=":", alpha=0.7,
             label="Kd raw NN")
    ax2.set_ylabel("Kd", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # Combined legend
    handles = [line_kp, line_ki, line_kd]
    labels = ["Kp", "Ki", "Kd"]
    ax.legend(handles, labels, loc="upper left", fontsize=9, ncol=3)

    # Annotate zone bands across the top of panel 4
    if zone_bands:
        ymax = ax.get_ylim()[1]
        for (t0, t1, lbl) in zone_bands:
            ax.text((t0 + t1) / 2, ymax * 0.97, lbl, ha="center", va="top",
                    fontsize=8, color="#444", alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff", ec="#ccc", lw=0.5))

    # KPI box on Panel 1
    settling = f"{kpis.settling_time:.2f} s" if not np.isnan(kpis.settling_time) else "--"
    text = (
        f"RMSE        : {kpis.rmse_velocity:.4f} m/s\n"
        f"IAE         : {kpis.iae_velocity:.3f}\n"
        f"Max |err|   : {kpis.max_abs_error:.4f} m/s\n"
        f"Overshoot   : {kpis.overshoot_pct:.2f} %\n"
        f"Settling    : {settling}\n"
        f"Compute     : {kpis.mean_compute_us:.1f} us/tick\n"
        f"Final s     : {kpis.final_position:.2f} m\n"
        f"Kp range    : [{np.nanmin(kp):.3f}, {np.nanmax(kp):.3f}]\n"
        f"Ki range    : [{np.nanmin(ki):.3f}, {np.nanmax(ki):.3f}]\n"
        f"Kd range    : [{np.nanmin(kd):.4f}, {np.nanmax(kd):.4f}]"
    )
    axes[0].text(
        0.015, 0.97, text,
        transform=axes[0].transAxes, fontsize=8,
        family="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#999", lw=0.7),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    if show:
        plt.show()
    plt.close(fig)
    return out_path.with_suffix(".png")
