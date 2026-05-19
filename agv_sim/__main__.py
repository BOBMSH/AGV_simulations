"""CLI entry point.

Usage:
    python -m agv_sim --scenario A --controller pid
    python -m agv_sim --scenario C --controllers pid,mpc      (comparison mode)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pathlib import Path
from .controllers import (MPCController, MPCParameters, PIDController, PIDGains,
                          IntelligentPIDController, IPIDConfig,
                          IntelligentMPCController, IMPCConfig)
from .plant import AGVParameters, AGVPlant
from .scenarios import SCENARIOS
from .utils import compute_kpis, run_simulation
from .viz import plot_comparison, plot_single_run


# Per-scenario default PID gains, tuned for the nominal plant.
DEFAULT_PID_GAINS = {
    "A": PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0),
    "B": PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0),  # default tuning
    "C": PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0),
    "D": PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0),
}


def build_controller(name: str, scenario):
    """Construct a controller for a given scenario.

    Scenario-aware: MPC uses Q_term_s>0 only when the scenario specifies a
    position-stop target (so it doesn't try to anchor at s=0 in scenarios
    without one).
    """
    if name == "pid":
        gains = DEFAULT_PID_GAINS.get(scenario.name, PIDGains(kp=0.8, ki=0.4, kd=0.05))
        return PIDController(gains)
    if name == "ipid":
        ckpt = Path(__file__).resolve().parent / "nn" / "checkpoints" / "ipid_scheduler.npz"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"IPID checkpoint not found at {ckpt}. "
                "Train it first: python -m agv_sim.nn.training"
            )
        return IntelligentPIDController(IPIDConfig(model_path=ckpt, smoothing_alpha=1.0))
    if name == "mpc":
        Q_term_s = 0.0
        if scenario.constraints().s_stop is not None:
            Q_term_s = 0.5
        return MPCController(MPCParameters(Q_term_s=Q_term_s))
    if name == "impc":
        ckpt = Path(__file__).resolve().parent / "nn" / "checkpoints" / "impc_residual.npz"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"IMPC residual model not found at {ckpt}. "
                "Train it first: python -m agv_sim.nn.impc_training"
            )
        Q_term_s = 0.0
        if scenario.constraints().s_stop is not None:
            Q_term_s = 0.5
        return IntelligentMPCController(IMPCConfig(
            residual_model_path=ckpt,
            mpc_params=MPCParameters(Q_term_s=Q_term_s),
        ))
    raise ValueError(f"Unknown controller {name!r}. Available: pid, ipid, mpc, impc")


def run_one(scenario, controller_name, params, seed):
    rng = np.random.default_rng(seed)
    plant = AGVPlant(params, scenario.environment(), rng=rng)
    controller = build_controller(controller_name, scenario)
    log = run_simulation(plant, controller, scenario, params, horizon=20)
    kpis = compute_kpis(
        t=log.t, v_ref=log.v_ref, v_actual=log.v_true,
        s_actual=log.s, u=log.u, compute_times_us=log.compute_us,
    )
    return log, kpis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AGV control benchmark")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS.keys()))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--controller", choices=["pid", "ipid", "mpc", "impc"],
                       help="Run a single controller (default: pid)")
    group.add_argument("--controllers", help="Comma-separated controllers to compare (e.g. pid,mpc)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    scenario_cls = SCENARIOS[args.scenario]
    scenario = scenario_cls()
    params = AGVParameters()

    # Resolve which controllers to run.
    if args.controllers:
        names = [c.strip() for c in args.controllers.split(",") if c.strip()]
    elif args.controller:
        names = [args.controller]
    else:
        names = ["pid"]

    print(f"[run] scenario={scenario.name}  controllers={names}  duration={scenario.duration:.1f}s")
    runs = []
    for cname in names:
        log, kpis = run_one(scenario, cname, params, args.seed)
        compute_disp = (
            f"{kpis.mean_compute_us:6.1f} us"
            if kpis.mean_compute_us < 1000.0
            else f"{kpis.mean_compute_us / 1000.0:6.2f} ms"
        )
        print(
            f"  {cname.upper():<5}  RMSE_v={kpis.rmse_velocity:.4f}  "
            f"IAE={kpis.iae_velocity:6.3f}  overshoot={kpis.overshoot_pct:5.2f}%  "
            f"effort={kpis.control_effort:6.3f}  compute={compute_disp}/tick  "
            f"final_s={kpis.final_position:7.3f}"
        )
        runs.append((cname, log, kpis))

    results_dir = Path(args.results_dir)
    if len(runs) == 1:
        cname, log, kpis = runs[0]
        fname = results_dir / f"scenario_{scenario.name}_{cname}"
        png = plot_single_run(
            log=log, kpis=kpis, controller_name=cname,
            scenario_name=scenario.name, scenario_description=scenario.description,
            out_path=fname, show=args.show,
        )
        print(f"[saved] {png}")
    else:
        names_joined = "_".join(n for n, _, _ in runs)
        fname = results_dir / f"scenario_{scenario.name}_compare_{names_joined}"
        png = plot_comparison(
            runs=runs,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            out_path=fname,
            v_max_fn=scenario.speed_limit_at,
            s_target=scenario.constraints().s_stop,
            s_tol=scenario.constraints().s_stop_tolerance,
            show=args.show,
        )
        print(f"[saved] {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
