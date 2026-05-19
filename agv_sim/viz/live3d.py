"""3D live animation: isometric AGV view + velocity + error panels.

3D path on the left, velocity + error stacked panels on the right, banner at
the bottom with live metrics.

CLI:
    python -m agv_sim.viz.live3d --scenario D --controller impc --save results/live3d_D_impc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ..plant import AGVParameters, AGVPlant
from ..scenarios import SCENARIOS
from ..utils.runner import run_simulation
from .live import _build_controller, _path_segments
from .static import CONTROLLER_COLOURS


def _agv_box(x: float, y0: float = 0.0, z_base: float = 0.0,
             length: float = 3.0, width: float = 1.6, height: float = 0.9,
             cab_h: float = 0.6) -> List[List[Tuple[float, float, float]]]:
    x0 = x - length / 2.0
    x1 = x + length / 2.0
    yA, yB = y0 - width / 2.0, y0 + width / 2.0
    zA, zB = z_base, z_base + height
    body = [
        [(x0, yA, zA), (x1, yA, zA), (x1, yB, zA), (x0, yB, zA)],
        [(x0, yA, zB), (x1, yA, zB), (x1, yB, zB), (x0, yB, zB)],
        [(x0, yA, zA), (x1, yA, zA), (x1, yA, zB), (x0, yA, zB)],
        [(x0, yB, zA), (x1, yB, zA), (x1, yB, zB), (x0, yB, zB)],
        [(x0, yA, zA), (x0, yB, zA), (x0, yB, zB), (x0, yA, zB)],
        [(x1, yA, zA), (x1, yB, zA), (x1, yB, zB), (x1, yA, zB)],
    ]
    cx0 = x + 0.05 * length
    cx1 = x + 0.45 * length
    cyA, cyB = y0 - 0.35 * width, y0 + 0.35 * width
    czA, czB = zB, zB + cab_h
    cab = [
        [(cx0, cyA, czA), (cx1, cyA, czA), (cx1, cyB, czA), (cx0, cyB, czA)],
        [(cx0, cyA, czB), (cx1, cyA, czB), (cx1, cyB, czB), (cx0, cyB, czB)],
        [(cx0, cyA, czA), (cx1, cyA, czA), (cx1, cyA, czB), (cx0, cyA, czB)],
        [(cx0, cyB, czA), (cx1, cyB, czA), (cx1, cyB, czB), (cx0, cyB, czB)],
        [(cx0, cyA, czA), (cx0, cyB, czA), (cx0, cyB, czB), (cx0, cyA, czB)],
        [(cx1, cyA, czA), (cx1, cyB, czA), (cx1, cyB, czB), (cx1, cyA, czB)],
    ]
    return body + cab


def _face_shades(base_hex: str) -> List[Tuple[float, float, float]]:
    rgb = np.array(mcolors.to_rgb(base_hex))
    mults = [0.55, 1.05, 0.85, 0.65, 0.75, 0.95,
             0.65, 1.15, 0.95, 0.75, 0.85, 1.05]
    return [tuple(np.clip(rgb * m, 0.0, 1.0)) for m in mults]


def _floor_tile_lines(s_max: float, y_half: float = 1.6, spacing: float = 2.0):
    return [[(x, -y_half, 0.005), (x, y_half, 0.005)]
            for x in np.arange(0, s_max + spacing, spacing)]


def make_animation_3d(
    scenario_name: str,
    controller_name: str,
    save_path: Optional[Path] = None,
    fps: int = 15,
    speedup: float = 5.0,
    seed: int = 0,
):
    scenario = SCENARIOS[scenario_name]()
    params = AGVParameters()
    ctrl = _build_controller(controller_name, scenario)
    plant = AGVPlant(params, scenario.environment(), rng=np.random.default_rng(seed))
    log = run_simulation(plant, ctrl, scenario, params, horizon=20)

    dt_anim = (1.0 / fps) * speedup
    frame_idx = np.unique(np.clip(
        np.round(np.arange(0, log.t[-1], dt_anim) / params.dt_ctrl).astype(int),
        0, len(log.t) - 1,
    ))
    n_frames = len(frame_idx)

    route_length = float(log.s.max())
    segments = _path_segments(scenario_name, route_length)
    base_colour = CONTROLLER_COLOURS.get(controller_name, "#1f77b4")
    face_cols = _face_shades(base_colour)

    # --- Figure: manual positioning to give the 3D scene most of the canvas ---
    fig = plt.figure(figsize=(16, 7), facecolor="#fafafa")

    # 3D axes: roughly 62% of width, almost full height
    ax_path = fig.add_axes([0.01, 0.10, 0.62, 0.82], projection="3d")
    # Right column 2D panels
    ax_v = fig.add_axes([0.69, 0.55, 0.29, 0.36])
    ax_err = fig.add_axes([0.69, 0.13, 0.29, 0.36])
    # Banner across the bottom
    ax_banner = fig.add_axes([0.05, 0.02, 0.90, 0.06])
    ax_banner.axis("off")

    # ---- 3D scene styling ----
    ax_path.set_facecolor("#fafafa")
    for pane in (ax_path.xaxis.pane, ax_path.yaxis.pane, ax_path.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))     # transparent walls
        pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    ax_path.grid(False)

    y_half = 1.6
    z_floor = 0.0
    for s0, s1, col, label in segments:
        ax_path.add_collection3d(Poly3DCollection(
            [[(s0, -y_half, z_floor), (s1, -y_half, z_floor),
              (s1, y_half, z_floor), (s0, y_half, z_floor)]],
            facecolors=col, alpha=0.55, edgecolors="#aaaaaa", linewidths=0.4,
        ))
    ax_path.add_collection3d(Line3DCollection(
        _floor_tile_lines(route_length, y_half), colors="white", linewidths=0.9, alpha=0.55,
    ))

    if scenario.constraints().s_stop is not None:
        sx = scenario.constraints().s_stop
        ax_path.add_collection3d(Poly3DCollection(
            [[(sx, -y_half, 0), (sx, y_half, 0), (sx, y_half, 1.9), (sx, -y_half, 1.9)]],
            facecolors="#8b0000", alpha=0.35, edgecolors="#5a0000", linewidths=0.6,
        ))
    if scenario_name == "D" and hasattr(scenario, "DESCENT_VMAX") and scenario.DESCENT_VMAX > 0.0:
        ds, de = scenario.DESCENT_START, scenario.DESCENT_END
        ax_path.add_collection3d(Poly3DCollection(
            [[(ds, -y_half, 0.01), (de, -y_half, 0.01),
              (de, y_half, 0.01), (ds, y_half, 0.01)]],
            facecolors="#ffd54f", alpha=0.45, edgecolors="#cc9000", linewidths=0.6,
        ))

    agv_poly = Poly3DCollection(_agv_box(log.s[0]), facecolors=face_cols,
                                edgecolors="#111", linewidths=0.8)
    ax_path.add_collection3d(agv_poly)
    trail_line, = ax_path.plot([log.s[0]], [0.0], [0.06],
                                color=base_colour, lw=2.8, alpha=0.85)

    ax_path.view_init(elev=20, azim=-58)
    ax_path.set_box_aspect((10, 1.4, 1.6))
    ax_path.set_xlim(-1.5, route_length + 1.5)
    ax_path.set_ylim(-y_half - 0.3, y_half + 0.3)
    ax_path.set_zlim(0, 2.4)
    # Clean axes: no internal ticks/labels (we'll annotate manually)
    ax_path.set_xticks([])
    ax_path.set_yticks([])
    ax_path.set_zticks([])
    ax_path.set_xlabel(""); ax_path.set_ylabel(""); ax_path.set_zlabel("")
    ax_path.set_title(
        f"Scenario {scenario.name}   ·   {controller_name.upper()}",
        fontsize=13, fontweight="bold", pad=4,
    )

    # Position scale ticks placed manually in 3D space (just under the floor)
    for x_tick in np.linspace(0, route_length, 5):
        ax_path.text(x_tick, -y_half - 0.5, 0, f"{x_tick:.0f} m",
                      fontsize=8, color="#444", ha="center", va="top")

    # Legend as figure-level patches (not inside 3D axes)
    legend_handles = []
    for s0, s1, col, label in segments:
        legend_handles.append(mpatches.Patch(color=col, alpha=0.55, label=label))
    if scenario.constraints().s_stop is not None:
        legend_handles.append(mpatches.Patch(color="#8b0000", alpha=0.35, label="dock face"))
    if scenario_name == "D" and hasattr(scenario, "DESCENT_VMAX") and scenario.DESCENT_VMAX > 0.0:
        legend_handles.append(mpatches.Patch(color="#ffd54f", alpha=0.45,
                                              label="speed-limit zone"))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper left",
                    bbox_to_anchor=(0.02, 1.0), fontsize=8, ncol=min(4, len(legend_handles)),
                    framealpha=0.85)

    # ---- Velocity panel ----
    ax_v.plot(log.t, log.v_ref, color="#666", lw=1.4, ls="--", label="v_ref")
    v_max_arr = np.array([scenario.speed_limit_at(float(s)) for s in log.s])
    if v_max_arr.max() < scenario.constraints().v_max - 1e-3:
        ax_v.plot(log.t, v_max_arr, color="#c54a4a", lw=0.9, ls=":", label="v_max(s)")
    v_line, = ax_v.plot([], [], color=base_colour, lw=1.8, label="v_true")
    ax_v.set_xlim(0, log.t[-1])
    ax_v.set_ylim(-0.05, max(log.v_ref.max(), log.v_true.max()) * 1.20 + 0.05)
    ax_v.set_ylabel("Velocity [m/s]")
    ax_v.set_title("Live velocity tracking", fontsize=10, loc="left", fontweight="bold")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="upper right", fontsize=8)

    # ---- Error panel ----
    err_full = log.v_ref - log.v_true
    ax_err.axhline(0.0, color="#444", lw=0.6)
    err_line, = ax_err.plot([], [], color=base_colour, lw=1.5)
    ax_err.set_xlim(0, log.t[-1])
    pad = max(0.1, np.abs(err_full).max() * 1.15)
    ax_err.set_ylim(-pad, pad)
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("v_ref − v_true [m/s]")
    ax_err.set_title("Live tracking error", fontsize=10, loc="left", fontweight="bold")
    ax_err.grid(True, alpha=0.3)

    banner = ax_banner.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#999", lw=0.7),
    )

    def update(fi):
        idx = int(frame_idx[fi])
        s_cur = float(log.s[idx])
        v_cur = float(log.v_true[idx])
        agv_poly.set_verts(_agv_box(s_cur))
        agv_poly.set_facecolors(face_cols)
        trail_line.set_data_3d(log.s[:idx + 1], np.zeros(idx + 1),
                                0.06 * np.ones(idx + 1))
        v_line.set_data(log.t[:idx + 1], log.v_true[:idx + 1])
        err_line.set_data(log.t[:idx + 1], err_full[:idx + 1])
        rmse_now = float(np.sqrt(np.mean(err_full[:idx + 1] ** 2)))
        banner.set_text(
            f"t = {log.t[idx]:5.2f} s    s = {s_cur:6.2f} m    "
            f"v = {v_cur:.3f} m/s    v_ref = {log.v_ref[idx]:.3f} m/s    "
            f"u = {log.u[idx]:+.3f}    RMSE so far = {rmse_now:.4f}"
        )
        return [agv_poly, trail_line, v_line, err_line, banner]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps, blit=False, repeat=False,
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
            anim.save(save_path.with_suffix(".mp4"), writer=writer, dpi=110)
            saved = save_path.with_suffix(".mp4")
        except (RuntimeError, FileNotFoundError):
            anim.save(save_path.with_suffix(".gif"),
                      writer=animation.PillowWriter(fps=fps), dpi=80)
            saved = save_path.with_suffix(".gif")
        print(f"[saved] {saved}")
        plt.close(fig)
        return saved
    plt.show()
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS.keys()))
    p.add_argument("--controller", required=True,
                    choices=["pid", "ipid", "mpc", "impc"])
    p.add_argument("--save", default=None)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--speedup", type=float, default=5.0)
    args = p.parse_args(argv)
    make_animation_3d(args.scenario, args.controller,
                       save_path=Path(args.save) if args.save else None,
                       fps=args.fps, speedup=args.speedup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
