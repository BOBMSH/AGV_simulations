"""4-up comparison animation: PID, IPID, MPC, IMPC on one scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..plant import AGVParameters, AGVPlant
from ..scenarios import SCENARIOS
from ..utils.runner import run_simulation
from .live import _build_controller, _path_segments
from .static import CONTROLLER_COLOURS


def make_comparison_animation(
    scenario_name: str,
    controllers: List[str],
    save_path: Optional[Path] = None,
    fps: int = 20,
    speedup: float = 2.5,
    seed: int = 0,
):
    """Render an animation with one row per controller, sharing time axis."""
    scenario = SCENARIOS[scenario_name]()
    params = AGVParameters()

    logs = []
    for cname in controllers:
        plant = AGVPlant(params, scenario.environment(),
                         rng=np.random.default_rng(seed))
        ctrl = _build_controller(cname, scenario)
        logs.append(run_simulation(plant, ctrl, scenario, params, horizon=20))

    n = len(controllers)
    dt_anim = (1.0 / fps) * speedup
    n_steps = min(len(l.t) for l in logs)
    frame_idx = np.unique(np.clip(
        np.round(np.arange(0, logs[0].t[n_steps - 1], dt_anim) / params.dt_ctrl).astype(int),
        0, n_steps - 1,
    ))
    n_frames = len(frame_idx)

    route_length = max(l.s.max() for l in logs)
    segments = _path_segments(scenario_name, route_length)

    fig = plt.figure(figsize=(13, 1.6 * n + 2.6))
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1] * n + [0.6],
                          width_ratios=[1.6, 1])
    ax_paths = [fig.add_subplot(gs[i, 0]) for i in range(n)]
    ax_errs = [fig.add_subplot(gs[i, 1]) for i in range(n)]
    ax_banner = fig.add_subplot(gs[n, :])
    ax_banner.axis("off")

    err_full_arr = [l.v_ref - l.v_true for l in logs]
    rmses_full = [float(np.sqrt(np.mean(e ** 2))) for e in err_full_arr]
    err_pad = max(0.1, max(np.abs(e).max() for e in err_full_arr) * 1.1)

    bodies, trails, err_lines = [], [], []
    for i, cname in enumerate(controllers):
        c = CONTROLLER_COLOURS.get(cname, "#222")
        ax_p = ax_paths[i]
        for s0, s1, col, label in segments:
            ax_p.add_patch(mpatches.Rectangle((s0, -0.5), s1 - s0, 1.0,
                                               facecolor=col, alpha=0.6))
        if scenario.constraints().s_stop is not None:
            ax_p.axvline(scenario.constraints().s_stop, color="#8b0000",
                         lw=1.4, ls="--")
        ax_p.set_xlim(-1, route_length + 2)
        ax_p.set_ylim(-1.2, 1.2)
        ax_p.set_yticks([])
        ax_p.set_ylabel(cname.upper(), fontsize=10, fontweight="bold", color=c)
        if i < n - 1:
            ax_p.set_xticklabels([])
        body = mpatches.Rectangle((-0.6, -0.3), 1.2, 0.6,
                                    facecolor=c, edgecolor="#222", lw=1)
        ax_p.add_patch(body)
        bodies.append(body)
        trail, = ax_p.plot([], [], color=c, lw=1.2, alpha=0.5)
        trails.append(trail)

        ax_e = ax_errs[i]
        ax_e.axhline(0.0, color="#444", lw=0.6)
        line, = ax_e.plot([], [], color=c, lw=1.2)
        ax_e.set_xlim(0, logs[0].t[n_steps - 1])
        ax_e.set_ylim(-err_pad, err_pad)
        if i < n - 1:
            ax_e.set_xticklabels([])
        if i == 0:
            ax_e.set_title("Tracking error (live)", fontsize=10)
        ax_e.grid(True, alpha=0.3)
        err_lines.append(line)

    ax_paths[0].set_title(f"Scenario {scenario_name} - 4-way comparison",
                           fontsize=11, fontweight="bold")
    ax_paths[-1].set_xlabel("Position s [m]")
    ax_errs[-1].set_xlabel("Time [s]")

    banner = ax_banner.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#999"),
    )

    def update(fi):
        idx = frame_idx[fi]
        lines = ""
        artists = []
        for i, cname in enumerate(controllers):
            log = logs[i]
            s_cur = log.s[idx]; v_cur = log.v_true[idx]
            bodies[i].set_xy((s_cur - 0.6, -0.3))
            trails[i].set_data(log.s[:idx + 1], np.zeros(idx + 1))
            err_lines[i].set_data(log.t[:idx + 1], err_full_arr[i][:idx + 1])
            rmse_so_far = float(np.sqrt(np.mean(err_full_arr[i][:idx + 1] ** 2)))
            lines += f"{cname.upper():<5}  s={s_cur:6.2f}  v={v_cur:.3f}  RMSE={rmse_so_far:.4f}    "
            artists += [bodies[i], trails[i], err_lines[i]]
        banner.set_text(f"t={logs[0].t[idx]:5.2f}s   |   " + lines.strip())
        artists.append(banner)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=1000.0 / fps, blit=True, repeat=False)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
            anim.save(save_path.with_suffix(".mp4"), writer=writer, dpi=100)
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
    p.add_argument("--controllers", default="pid,ipid,mpc,impc")
    p.add_argument("--save", default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--speedup", type=float, default=2.5)
    args = p.parse_args(argv)
    names = [c.strip() for c in args.controllers.split(",") if c.strip()]
    save = Path(args.save) if args.save else None
    make_comparison_animation(args.scenario, names, save_path=save,
                                fps=args.fps, speedup=args.speedup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
