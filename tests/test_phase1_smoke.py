"""Phase-1 smoke tests.

Plain-Python asserts (no pytest dependency). Run with:
    python -m tests.test_phase1_smoke
"""

from __future__ import annotations

import numpy as np

from agv_sim.controllers import PIDController, PIDGains
from agv_sim.plant import AGVParameters, AGVPlant, EnvironmentProfile
from agv_sim.scenarios import ScenarioA
from agv_sim.utils import compute_kpis, run_simulation


def test_plant_at_rest_stays_at_rest() -> None:
    p = AGVParameters()
    plant = AGVPlant(p, EnvironmentProfile(), rng=np.random.default_rng(0))
    plant.reset()
    for _ in range(100):
        plant.step(0.0, p.dt_sim)
    assert plant.state.v == 0.0, f"plant moved without input: v={plant.state.v}"


def test_plant_full_throttle_saturates_at_vmax() -> None:
    p = AGVParameters()
    plant = AGVPlant(p, EnvironmentProfile(), rng=np.random.default_rng(0))
    plant.reset()
    for _ in range(2000):
        plant.step(1.0, p.dt_sim)
    assert abs(plant.state.v - p.v_max) < 1e-6, f"v_max not reached: {plant.state.v}"


def test_pid_reset_clears_state() -> None:
    pid = PIDController(PIDGains(kp=1.0, ki=0.5, kd=0.1))
    pid.step(0.0, 1.0, 0.05)
    assert pid._integral != 0.0
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error is None


def test_scenario_a_pid_meets_tracking_threshold() -> None:
    """End-to-end: PID on Scenario A must achieve RMSE < 0.05 m/s and final position > 30 m."""
    s = ScenarioA()
    p = AGVParameters()
    plant = AGVPlant(p, s.environment(), rng=np.random.default_rng(0))
    ctrl = PIDController(PIDGains(kp=0.85, ki=0.45, kd=0.05, n_d=15.0))
    log = run_simulation(plant, ctrl, s, p)
    kpis = compute_kpis(
        t=log.t, v_ref=log.v_ref, v_actual=log.v_true,
        s_actual=log.s, u=log.u, compute_times_us=log.compute_us,
    )
    assert kpis.rmse_velocity < 0.05, f"PID RMSE too high: {kpis.rmse_velocity}"
    assert kpis.final_position > 30.0, f"AGV did not cover the aisle: {kpis.final_position}"
    assert not np.isnan(kpis.settling_time), "settling time should be finite"


if __name__ == "__main__":
    test_plant_at_rest_stays_at_rest(); print("✓ plant at rest")
    test_plant_full_throttle_saturates_at_vmax(); print("✓ plant saturates at v_max")
    test_pid_reset_clears_state(); print("✓ PID reset clears state")
    test_scenario_a_pid_meets_tracking_threshold(); print("✓ Scenario A end-to-end")
    print("All Phase-1 smoke tests passed.")
