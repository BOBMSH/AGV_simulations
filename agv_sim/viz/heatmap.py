"""Cross-scenario summary heatmap.

Renders a publication-quality figure with three side-by-side panels:
  (1) RMSE_v heatmap (lower is better)
  (2) Mean compute-time heatmap (lower is better; log-coloured)
  (3) Engineered-winner table + commentary
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from ..scenarios import SCENARIOS
from ..utils.sweep import SweepResult


# Per-scenario "true winning metric". RMSE is the natural one for A/B/D;
# Scenario C's win condition is position-stop tolerance.
WINNING_METRIC = {
    "A": ("compute", "lowest compute per tick at competitive RMSE"),
    "B": ("rmse_velocity", "lowest velocity RMSE under varying surface+payload"),
    "C": ("dock_tolerance", "parks within +/-5 cm of the dock"),
    "D": ("rmse_velocity", "lowest velocity RMSE on the combined disturbance"),
}


def _winning_summary(result: SweepResult) -> str:
    lines = []
    for s in result.scenarios:
        fav = SCENARIOS[s]().favored_controller
        metric, blurb = WINNING_METRIC.get(s, ("rmse_velocity", "lowest RMSE"))
        kfav = result.kpis[s][fav]
        if metric == "compute":
            cdisp = (f"{kfav.mean_compute_us:.1f} us"
                     if kfav.mean_compute_us < 1000.0
                     else f"{kfav.mean_compute_us/1000.0:.2f} ms")
            lines.append(f"  Scenario {s}: {fav.upper()} wins on  {blurb}.\n"
                         f"               compute = {cdisp}/tick.")
        elif metric == "dock_tolerance":
            err = kfav.final_position - SCENARIOS[s]().constraints().s_stop
            tol = SCENARIOS[s]().constraints().s_stop_tolerance
            ok = "PASS" if abs(err) <= tol else "FAIL"
            lines.append(f"  Scenario {s}: {fav.upper()} wins on  {blurb}.\n"
                         f"               err = {err*100:+.1f} cm  [{ok}]")
        else:
            lines.append(f"  Scenario {s}: {fav.upper()} wins on  {blurb}.\n"
                         f"               RMSE = {kfav.rmse_velocity:.4f} m/s")
    return "\n".join(lines)


def render_summary(result: SweepResult, out_path: Path, show: bool = False) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_s = len(result.scenarios)
    n_c = len(result.controllers)
    rmse = result.matrix("rmse_velocity")
    compute = result.matrix("mean_compute_us")
    favoured = [SCENARIOS[s]().favored_controller for s in result.scenarios]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [1, 1]})

    # --- Panel 1: RMSE heatmap ----------------------------------------------
    ax = axes[0]
    im = ax.imshow(rmse, cmap="RdYlGn_r", aspect="auto",
                    norm=mcolors.Normalize(vmin=rmse.min(), vmax=rmse.max()))
    for i in range(n_s):
        for j in range(n_c):
            v = rmse[i, j]
            colour = "white" if v > (rmse.min() + rmse.max()) / 2 else "#222"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=10, color=colour, fontweight="bold")
    for i, fav in enumerate(favoured):
        if fav in result.controllers:
            j = result.controllers.index(fav)
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="#003a8c",
                                       lw=2.5, ls="--"))
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([c.upper() for c in result.controllers], fontsize=10)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([f"Scenario {s}" for s in result.scenarios], fontsize=10)
    ax.set_title("Velocity tracking RMSE  (m/s)\nlower is better  |  dashed = engineered winner",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Panel 2: Compute heatmap ------------------------------------------
    ax = axes[1]
    log_compute = np.log10(np.maximum(compute, 1.0))
    im2 = ax.imshow(log_compute, cmap="Reds", aspect="auto")
    for i in range(n_s):
        for j in range(n_c):
            v = compute[i, j]
            disp = f"{v:.1f} us" if v < 1000.0 else f"{v/1000.0:.2f} ms"
            mid = (log_compute.min() + log_compute.max()) / 2
            colour = "white" if log_compute[i, j] > mid else "#222"
            ax.text(j, i, disp, ha="center", va="center", fontsize=10,
                    color=colour, fontweight="bold")
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([c.upper() for c in result.controllers], fontsize=10)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([f"Scenario {s}" for s in result.scenarios], fontsize=10)
    ax.set_title("Mean compute per controller tick\nlower is better  |  log colour scale",
                 fontsize=11, fontweight="bold")

    fig.suptitle("AGV Control Strategy Benchmark - Cross-Scenario Summary",
                 fontsize=13, fontweight="bold", y=1.02)

    # Annotation box at the bottom with per-scenario winning narrative.
    summary_text = _winning_summary(result)
    fig.text(
        0.5, -0.04, summary_text, ha="center", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#f4f4f4",
                  ec="#999", lw=0.7),
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path.with_suffix(".png")
