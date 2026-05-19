"""Phase-3 smoke tests.

Run: python -m tests.test_phase3_smoke
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agv_sim.controllers import (
    IntelligentPIDController, IPIDConfig,
    MPCController, MPCParameters,
    PIDController, PIDGains,
)
from agv_sim.nn import MLP, MLPConfig
from agv_sim.plant import AGVParameters, AGVPlant
from agv_sim.scenarios import ScenarioB
from agv_sim.utils import run_simulation


CKPT = Path(__file__).resolve().parents[1] / "agv_sim" / "nn" / "checkpoints" / "ipid_scheduler.npz"


def _run(ctrl, scenario, params):
    plant = AGVPlant(params, scenario.environment(), rng=np.random.default_rng(0))
    return run_simulation(plant, ctrl, scenario, params, horizon=20)


def test_mlp_save_load_roundtrip() -> None:
    """MLP save/load is bit-exact."""
    mlp = MLP(MLPConfig(layer_sizes=(2, 8, 4), out_activation="softplus", seed=0))
    X = np.random.default_rng(0).uniform(0, 1, (20, 2))
    Y = np.random.default_rng(1).uniform(0.1, 1.0, (20, 4))
    mlp.fit(X, Y, n_epochs=50, normalise_x=True, normalise_y=False, verbose=False)
    mlp.save("/tmp/test_mlp_roundtrip.npz")
    mlp2 = MLP.load("/tmp/test_mlp_roundtrip.npz")
    x = np.array([0.3, 0.6])
    assert np.allclose(mlp.forward(x), mlp2.forward(x), atol=1e-12)


def test_ipid_checkpoint_exists_and_loads() -> None:
    """IPID checkpoint must exist and load cleanly."""
    assert CKPT.exists(), f"IPID checkpoint not found at {CKPT}; run python -m agv_sim.nn.training"
    ctrl = IntelligentPIDController(IPIDConfig(model_path=CKPT, smoothing_alpha=1.0))
    # Gains should differ across operating points.
    g_low = ctrl._infer_gains({"surface_cr": 0.005, "payload_kg": 0.0})
    g_high = ctrl._infer_gains({"surface_cr": 0.030, "payload_kg": 14000.0})
    assert g_low is not None and g_high is not None
    # Higher Cr -> higher Kp
    assert g_high[0] > g_low[0], f"IPID gains not adapting upward with Cr: low={g_low[0]} high={g_high[0]}"


def test_ipid_beats_pid_on_scenario_B() -> None:
    """On Scenario B, IPID's velocity-tracking RMSE should be below baseline PID."""
    sc = ScenarioB()
    p = AGVParameters()
    pid = PIDController(PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0))
    ipid = IntelligentPIDController(IPIDConfig(model_path=CKPT, smoothing_alpha=1.0))
    log_pid = _run(pid, sc, p)
    log_ipid = _run(ipid, sc, p)
    rmse_pid = float(np.sqrt(np.mean((log_pid.v_ref - log_pid.v_true) ** 2)))
    rmse_ipid = float(np.sqrt(np.mean((log_ipid.v_ref - log_ipid.v_true) ** 2)))
    assert rmse_ipid < rmse_pid, f"IPID did not beat PID on B: PID={rmse_pid:.4f}, IPID={rmse_ipid:.4f}"


def test_ipid_compute_remains_microseconds() -> None:
    """IPID's per-tick compute should stay in microseconds (PID-class), not milliseconds."""
    sc = ScenarioB()
    p = AGVParameters()
    ipid = IntelligentPIDController(IPIDConfig(model_path=CKPT, smoothing_alpha=1.0))
    log = _run(ipid, sc, p)
    median_us = float(np.median(log.compute_us))
    assert median_us < 500.0, f"IPID compute too slow: median {median_us:.0f} us"


if __name__ == "__main__":
    test_mlp_save_load_roundtrip(); print("OK MLP save/load roundtrip")
    test_ipid_checkpoint_exists_and_loads(); print("OK IPID checkpoint loads, gains adapt")
    test_ipid_beats_pid_on_scenario_B(); print("OK IPID beats PID on Scenario B")
    test_ipid_compute_remains_microseconds(); print("OK IPID compute stays in microseconds")
    print("All Phase-3 smoke tests passed.")
