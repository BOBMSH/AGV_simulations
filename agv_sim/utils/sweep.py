"""Cross-scenario, cross-controller KPI sweep.

Runs all (controller, scenario) combinations and aggregates KPIs into a
single rectangular array suitable for heatmap rendering and CSV export.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..controllers import (
    IntelligentMPCController, IMPCConfig,
    IntelligentPIDController, IPIDConfig,
    MPCController, MPCParameters,
    PIDController, PIDGains,
)
from ..plant import AGVParameters, AGVPlant
from ..scenarios import SCENARIOS
from .kpi import KPIs, compute_kpis
from .runner import run_simulation


CONTROLLER_NAMES = ["pid", "ipid", "mpc", "impc"]
SCENARIO_NAMES = ["A", "B", "C", "D"]


def _build_controller(name: str, scenario, ckpt_dir: Path):
    if name == "pid":
        return PIDController(PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0))
    if name == "ipid":
        return IntelligentPIDController(IPIDConfig(
            model_path=ckpt_dir / "ipid_scheduler.npz",
            smoothing_alpha=1.0,
        ))
    if name == "mpc":
        Q_term_s = 0.5 if scenario.constraints().s_stop is not None else 0.0
        return MPCController(MPCParameters(Q_term_s=Q_term_s))
    if name == "impc":
        Q_term_s = 0.5 if scenario.constraints().s_stop is not None else 0.0
        return IntelligentMPCController(IMPCConfig(
            residual_model_path=ckpt_dir / "impc_residual.npz",
            mpc_params=MPCParameters(Q_term_s=Q_term_s),
        ))
    raise ValueError(f"unknown controller {name!r}")


@dataclass
class SweepResult:
    """Aggregated KPIs for all (controller, scenario) combos."""

    controllers: List[str]
    scenarios: List[str]
    kpis: Dict[str, Dict[str, KPIs]]   # kpis[scenario][controller] = KPIs

    def matrix(self, kpi_name: str) -> np.ndarray:
        """Return a (n_scenarios, n_controllers) matrix of the given KPI."""
        n_s = len(self.scenarios)
        n_c = len(self.controllers)
        M = np.full((n_s, n_c), np.nan)
        for i, sc in enumerate(self.scenarios):
            for j, ctrl in enumerate(self.controllers):
                k = self.kpis.get(sc, {}).get(ctrl)
                if k is not None:
                    M[i, j] = getattr(k, kpi_name)
        return M


def run_sweep(
    seed: int = 0,
    ckpt_dir: Path = None,
    verbose: bool = True,
) -> SweepResult:
    """Run every (controller, scenario) pair. Returns a SweepResult."""
    if ckpt_dir is None:
        ckpt_dir = Path(__file__).resolve().parents[1] / "nn" / "checkpoints"
    params = AGVParameters()
    kpis: Dict[str, Dict[str, KPIs]] = {}
    for sname in SCENARIO_NAMES:
        scenario = SCENARIOS[sname]()
        kpis[sname] = {}
        for cname in CONTROLLER_NAMES:
            ctrl = _build_controller(cname, scenario, ckpt_dir)
            plant = AGVPlant(params, scenario.environment(),
                             rng=np.random.default_rng(seed))
            log = run_simulation(plant, ctrl, scenario, params, horizon=20)
            k = compute_kpis(
                t=log.t, v_ref=log.v_ref, v_actual=log.v_true,
                s_actual=log.s, u=log.u, compute_times_us=log.compute_us,
            )
            kpis[sname][cname] = k
            if verbose:
                cdisp = (f"{k.mean_compute_us:6.1f} us" if k.mean_compute_us < 1000.0
                         else f"{k.mean_compute_us/1000.0:6.2f} ms")
                print(f"  scenario {sname}  {cname.upper():<5}  RMSE={k.rmse_velocity:.4f}  "
                      f"IAE={k.iae_velocity:7.3f}  compute={cdisp}/tick  final_s={k.final_position:7.3f}")
    return SweepResult(
        controllers=CONTROLLER_NAMES,
        scenarios=SCENARIO_NAMES,
        kpis=kpis,
    )


def save_csv(result: SweepResult, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario", "controller",
        "rmse_velocity", "iae_velocity", "max_abs_error",
        "overshoot_pct", "control_effort", "settling_time",
        "mean_compute_us", "final_position",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for sc in result.scenarios:
            for ctrl in result.controllers:
                k = result.kpis[sc][ctrl]
                w.writerow([
                    sc, ctrl,
                    f"{k.rmse_velocity:.6f}", f"{k.iae_velocity:.6f}",
                    f"{k.max_abs_error:.6f}", f"{k.overshoot_pct:.4f}",
                    f"{k.control_effort:.6f}", f"{k.settling_time:.6f}",
                    f"{k.mean_compute_us:.3f}", f"{k.final_position:.4f}",
                ])




def load_csv(in_path: Path) -> SweepResult:
    """Reconstruct a SweepResult from a previously-saved CSV."""
    import csv as _csv
    kpis: Dict[str, Dict[str, KPIs]] = {}
    with open(in_path) as f:
        r = _csv.DictReader(f)
        for row in r:
            sc = row['scenario']
            c = row['controller']
            kpis.setdefault(sc, {})[c] = KPIs(
                rmse_velocity=float(row['rmse_velocity']),
                iae_velocity=float(row['iae_velocity']),
                max_abs_error=float(row['max_abs_error']),
                overshoot_pct=float(row['overshoot_pct']),
                control_effort=float(row['control_effort']),
                settling_time=float(row['settling_time']),
                mean_compute_us=float(row['mean_compute_us']),
                final_position=float(row['final_position']),
            )
    return SweepResult(
        controllers=CONTROLLER_NAMES,
        scenarios=SCENARIO_NAMES,
        kpis=kpis,
    )

if __name__ == "__main__":
    result = run_sweep(verbose=True)
    out = Path(__file__).resolve().parents[2] / "results" / "kpi_sweep.csv"
    save_csv(result, out)
    print(f"\n[saved] {out}")
