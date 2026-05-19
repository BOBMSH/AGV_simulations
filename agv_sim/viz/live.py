"""Live 2D animation of the AGV simulation.

Left panel: top-down view of the AGV moving along the scenario path. The
path is coloured to show surface zones (Scenario B), grade zones
(Scenario D), or the speed-limit funnel (Scenario C). The AGV is drawn
as a rectangle that moves along the path.

Right panel: tracking-error vs time, building in real time as the
animation progresses.

Bottom strip: KPI banner with the live RMSE, current u, and (for
intelligent controllers) the live NN-derived gains/residual.

Designed to be both runnable as a CLI ("python -m agv_sim.viz.live ...")
for live presentation, and exportable as an MP4 for safe playback.

The simulation runs OFFLINE first so the animation can be a clean
replay (no risk of solver hiccups during the talk).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..controllers import (
    IntelligentMPCController, IMPCConfig,
    IntelligentPIDController, IPIDConfig,
    MPCController, MPCParameters,
    PIDController, PIDGains,
)
from ..plant import AGVParameters, AGVPlant
from ..scenarios import SCENARIOS
from ..utils.runner import SimulationLog, run_simulation
from .static import CONTROLLER_COLOURS


def _build_controller(name: str, scenario):
    ck = Path(__file__).resolve().parents[1] / "nn" / "checkpoints"
    if name == "pid":
        return PIDController(PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0))
    if name == "ipid":
        return IntelligentPIDController(IPIDConfig(model_path=ck / "ipid_scheduler.npz",
                                                    smoothing_alpha=1.0))
    if name == "mpc":
        Q_term_s = 0.5 if scenario.constraints().s_stop is not None else 0.0
        return MPCController(MPCParameters(Q_term_s=Q_term_s))
    if name == "impc":
        Q_term_s = 0.5 if scenario.constraints().s_stop is not None else 0.0
        return IntelligentMPCController(IMPCConfig(
            residual_model_path=ck / "impc_residual.npz",
            mpc_params=MPCParameters(Q_term_s=Q_term_s),
        ))
    raise ValueError(f"unknown controller {name!r}")


def _path_segments(scenario_name: str, route_length: float) -> List[Tuple[float, float, str, str]]:
    """Return (s_start, s_end, colour, label) for visualising path zones."""
    if scenario_name == "A":
        return [(0.0, route_length, "#dddddd", "smooth concrete")]
    if scenario_name == "B":
        from ..scenarios.scenario_b_friction import SURFACES
        cmap = {0.004: "#dddddd", 0.012: "#bbbbcc", 0.022: "#90b0e0", 0.030: "#d2a26b"}
        return [(s.s_start, s.s_end, cmap.get(round(s.cr, 4), "#888"), s.label)
                for s in SURFACES]
    if scenario_name == "C":
        return [
            (0.0, 48.0, "#dddddd", "approach"),
            (48.0, 50.0, "#f5b8b8", "funnel (v_max ramps)"),
        ]
    if scenario_name == "D":
        return [
            (0.0, 20.0, "#dddddd", "flat (empty)"),
            (20.0, 50.0, "#9cb6dc", "6% wet uphill (loaded)"),
            (50.0, 55.0, "#cccccc", "crest"),
            (55.0, 80.0, "#a4d5a3", "4% descent"),
        ]
    return [(0.0, route_length, "#dddddd", "path")]


def make_animation(
    scenario_name: str,
    controller_name: str,
    save_path: Optional[Path] = None,
    fps: int = 30,
    speedup: float = 1.0,
    show: bool = False,
    seed: int = 0,
):
    """Produce an animation (and optionally save MP4) of one (scenario, controller) run."""
    scenario = SCENARIOS[scenario_name]()
    params = AGVParameters()
    ctrl = _build_controller(controller_name, scenario)
    plant = AGVPlant(params, scenario.environment(), rng=np.random.default_rng(seed))
    log = run_simulation(plant, ctrl, scenario, params, horizon=20)

    # Subsample to the desired animation frame rate.
    dt_anim = (1.0 / fps) * speedup
    frame_idx = np.unique(np.clip(
        np.round(np.arange(0, log.t[-1], dt_anim) / params.dt_ctrl).astype(int),
        0, len(log.t) - 1,
    ))
    n_frames = len(frame_idx)

    route_length = log.s.max()
    segments = _path_segments(scenario_name, route_length)
    colour = CONTROLLER_COLOURS.get(controller_name, "#222222")

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 0.5], width_ratios=[1.3, 1])
    ax_path = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_banner = fig.add_subplot(gs[1, :])
    ax_banner.axis("off")

    # --- Path panel ---
    for s0, s1, col, label in segments:
        ax_path.add_patch(mpatches.Rectangle(
            (s0, -0.5), s1 - s0, 1.0, facecolor=col, edgecolor="none", alpha=0.6,
            label=label,
        ))
    # Dock marker for Scenario C
    if scenario.constraints().s_stop is not None:
        ax_path.axvline(scenario.constraints().s_stop, color="#8b0000", lw=1.8, ls="--")
        tol = scenario.constraints().s_stop_tolerance
        ax_path.axvspan(scenario.constraints().s_stop - tol,
                         scenario.constraints().s_stop + tol,
                         color="#8b0000", alpha=0.15)
    ax_path.set_xlim(-1, route_length + 2)
    ax_path.set_ylim(-1.5, 1.5)
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.set_xlabel("Position s [m]")
    ax_path.set_yticks([])
    ax_path.set_title(f"Scenario {scenario_name}  -  {controller_name.upper()}",
                       fontsize=11, fontweight="bold")
    # Compact legend
    by_label = {}
    for h, l in zip(*ax_path.get_legend_handles_labels()):
        by_label[l] = h
    ax_path.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=7,
                    ncol=2, framealpha=0.85)
    # AGV body marker
    agv_body = mpatches.Rectangle((-0.6, -0.3), 1.2, 0.6,
                                    facecolor=colour, edgecolor="#222", lw=1.2)
    ax_path.add_patch(agv_body)
    # Trail
    trail_line, = ax_path.plot([], [], color=colour, lw=1.5, alpha=0.5)

    # --- Error panel ---
    ax_err.axhline(0.0, color="#444", lw=0.6)
    err_line, = ax_err.plot([], [], color=colour, lw=1.4)
    ax_err.set_xlim(0, log.t[-1])
    err_full = log.v_ref - log.v_true
    pad = max(0.1, np.abs(err_full).max() * 1.1)
    ax_err.set_ylim(-pad, pad)
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("v_ref - v_true [m/s]")
    ax_err.set_title("Tracking error (live)", fontsize=10)
    ax_err.grid(True, alpha=0.3)

    # --- Banner ---
    banner_text = ax_banner.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#999", lw=0.7),
    )

    def update(fi):
        idx = frame_idx[fi]
        s_cur = log.s[idx]
        v_cur = log.v_true[idx]
        agv_body.set_xy((s_cur - 0.6, -0.3))
        trail_line.set_data(log.s[:idx + 1], np.zeros(idx + 1))
        err_line.set_data(log.t[:idx + 1], err_full[:idx + 1])
        rmse_so_far = float(np.sqrt(np.mean(err_full[:idx + 1] ** 2)))
        banner_text.set_text(
            f"t={log.t[idx]:5.2f}s   s={s_cur:6.2f} m   v={v_cur:.3f} m/s   "
            f"v_ref={log.v_ref[idx]:.3f}   u={log.u[idx]:+.3f}   "
            f"RMSE so far = {rmse_so_far:.4f}"
        )
        return agv_body, trail_line, err_line, banner_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps, blit=True, repeat=False,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Try mp4 first; fall back to gif if ffmpeg unavailable.
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(save_path.with_suffix(".mp4"), writer=writer, dpi=110)
            saved = save_path.with_suffix(".mp4")
        except (RuntimeError, FileNotFoundError):
            anim.save(save_path.with_suffix(".gif"),
                      writer=animation.PillowWriter(fps=fps), dpi=90)
            saved = save_path.with_suffix(".gif")
        print(f"[saved] {saved}")
        plt.close(fig)
        return saved
    if show:
        plt.show()
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AGV live animation")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS.keys()))
    parser.add_argument("--controller", required=True,
                        choices=["pid", "ipid", "mpc", "impc"])
    parser.add_argument("--save", default=None, help="output MP4/GIF path (no ext needed)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speedup", type=float, default=2.0,
                        help="playback speedup (>1 plays scenario faster)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    save_path = Path(args.save) if args.save else None
    make_animation(args.scenario, args.controller, save_path=save_path,
                    fps=args.fps, speedup=args.speedup, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
