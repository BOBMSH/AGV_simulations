"""Phase-4 smoke tests.

Run: python -m tests.test_phase4_smoke
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agv_sim.controllers import (
    IntelligentMPCController, IMPCConfig,
    MPCController, MPCParameters,
    PIDController, PIDGains,
)
from agv_sim.nn import MLP
from agv_sim.plant import AGVParameters, AGVPlant
from agv_sim.scenarios import ScenarioD
from agv_sim.utils import run_simulation


CKPT = Path(__file__).resolve().parents[1] / "agv_sim" / "nn" / "checkpoints" / "impc_residual.npz"


def _run(ctrl, scenario, params):
    plant = AGVPlant(params, scenario.environment(), rng=np.random.default_rng(0))
    return run_simulation(plant, ctrl, scenario, params, horizon=20)


def test_impc_residual_checkpoint_exists() -> None:
    assert CKPT.exists(), f"IMPC residual model not found at {CKPT}"
    mlp = MLP.load(CKPT)
    assert mlp.cfg.layer_sizes == (6, 32, 32, 1)


def test_impc_jacobian_matches_finite_difference() -> None:
    """Analytical Jacobian should match finite differences to numerical precision."""
    mlp = MLP.load(CKPT)
    x = np.array([1.0, 0.5, 0.0, 0.012, 8000.0, 0.03])
    y_pred, jac = mlp.jacobian(x)
    h = 1e-4
    fd = np.zeros_like(jac)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        fd[:, i] = (mlp.forward(xp) - mlp.forward(xm)) / (2 * h)
    rel_err = np.abs(jac - fd).max() / (np.abs(jac).max() + 1e-12)
    assert rel_err < 1e-4, f"Jacobian disagrees with FD: rel_err={rel_err}"


def test_impc_qp_is_dpp() -> None:
    """The IMPC QP must be DPP-compliant despite the residual parameters."""
    ctrl = IntelligentMPCController(IMPCConfig(residual_model_path=CKPT))
    assert ctrl._problem.is_dpp()


def test_impc_beats_mpc_on_scenario_D() -> None:
    """On Scenario D, the NN-residual IMPC must track tighter than the
    fixed-plant Traditional MPC."""
    sc = ScenarioD()
    p = AGVParameters()
    mpc = MPCController(MPCParameters(Q_term_s=0.0))
    impc = IntelligentMPCController(IMPCConfig(
        residual_model_path=CKPT,
        mpc_params=MPCParameters(Q_term_s=0.0),
    ))
    log_mpc = _run(mpc, sc, p)
    log_impc = _run(impc, sc, p)
    rmse_mpc = float(np.sqrt(np.mean((log_mpc.v_ref - log_mpc.v_true) ** 2)))
    rmse_impc = float(np.sqrt(np.mean((log_impc.v_ref - log_impc.v_true) ** 2)))
    assert rmse_impc < rmse_mpc, (
        f"IMPC did not beat MPC on D: MPC RMSE={rmse_mpc:.4f}, IMPC RMSE={rmse_impc:.4f}"
    )


def test_impc_beats_pid_on_scenario_D() -> None:
    """IMPC should also beat baseline PID on Scenario D."""
    sc = ScenarioD()
    p = AGVParameters()
    pid = PIDController(PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0))
    impc = IntelligentMPCController(IMPCConfig(
        residual_model_path=CKPT,
        mpc_params=MPCParameters(Q_term_s=0.0),
    ))
    log_pid = _run(pid, sc, p)
    log_impc = _run(impc, sc, p)
    rmse_pid = float(np.sqrt(np.mean((log_pid.v_ref - log_pid.v_true) ** 2)))
    rmse_impc = float(np.sqrt(np.mean((log_impc.v_ref - log_impc.v_true) ** 2)))
    assert rmse_impc < rmse_pid, (
        f"IMPC did not beat PID on D: PID RMSE={rmse_pid:.4f}, IMPC RMSE={rmse_impc:.4f}"
    )


def test_impc_solve_time_within_tick_budget() -> None:
    sc = ScenarioD()
    p = AGVParameters()
    impc = IntelligentMPCController(IMPCConfig(residual_model_path=CKPT))
    log = _run(impc, sc, p)
    median_ms = float(np.median(log.compute_us)) / 1000.0
    assert median_ms < 30.0, f"IMPC compute too slow: median {median_ms:.2f} ms"


if __name__ == "__main__":
    test_impc_residual_checkpoint_exists(); print("OK IMPC residual checkpoint exists")
    test_impc_jacobian_matches_finite_difference(); print("OK Jacobian matches finite difference")
    test_impc_qp_is_dpp(); print("OK IMPC QP is DPP-compliant")
    test_impc_beats_mpc_on_scenario_D(); print("OK IMPC beats MPC on Scenario D")
    test_impc_beats_pid_on_scenario_D(); print("OK IMPC beats PID on Scenario D")
    test_impc_solve_time_within_tick_budget(); print("OK IMPC solve time within budget")
    print("All Phase-4 smoke tests passed.")
