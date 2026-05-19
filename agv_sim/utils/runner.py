"""Simulation runner.

Composes a plant, a scenario, and a controller, and runs them to completion.
Returns a dictionary of per-tick logs that can be fed into the KPI module or
the visualisation module.

The physics runs at dt_sim (default 100 Hz); the controller is invoked at
dt_ctrl (default 20 Hz). Between controller ticks the most recent control
input is held (zero-order hold) - standard for embedded control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from ..controllers.base import Controller
from ..plant import AGVParameters, AGVPlant
from ..scenarios.base import Scenario


@dataclass
class SimulationLog:
    """Per-tick log at the controller cadence."""

    t: np.ndarray
    v_ref: np.ndarray
    v_meas: np.ndarray
    v_true: np.ndarray
    s: np.ndarray
    u: np.ndarray
    compute_us: np.ndarray
    diagnostics: list


def run_simulation(
    plant: AGVPlant,
    controller: Controller,
    scenario: Scenario,
    params: AGVParameters,
    horizon: int = 20,
) -> SimulationLog:
    """Run one (plant, controller, scenario) simulation to scenario.duration."""
    plant.reset()
    controller.reset()

    dt_sim = params.dt_sim
    dt_ctrl = params.dt_ctrl
    n_substeps = max(1, int(round(dt_ctrl / dt_sim)))
    n_ctrl = int(round(scenario.duration / dt_ctrl))

    t_log = np.zeros(n_ctrl)
    v_ref_log = np.zeros(n_ctrl)
    v_meas_log = np.zeros(n_ctrl)
    v_true_log = np.zeros(n_ctrl)
    s_log = np.zeros(n_ctrl)
    u_log = np.zeros(n_ctrl)
    compute_us_log = np.zeros(n_ctrl)
    diag_log: List[dict] = []

    for k in range(n_ctrl):
        t_now = k * dt_ctrl
        v_ref = scenario.reference_velocity(t_now)
        v_meas = plant.measure()
        preview = scenario.preview(
            t_now, horizon, dt_ctrl, s_now=plant.state.s, v_now=v_meas
        )

        t0 = time.perf_counter()
        out = controller.step(v_meas, v_ref, dt_ctrl, preview=preview, s_meas=plant.state.s)
        compute_us = (time.perf_counter() - t0) * 1e6

        for _ in range(n_substeps):
            plant.step(out.u, dt_sim)

        t_log[k] = t_now
        v_ref_log[k] = v_ref
        v_meas_log[k] = v_meas
        v_true_log[k] = plant.state.v
        s_log[k] = plant.state.s
        u_log[k] = out.u
        compute_us_log[k] = compute_us
        diag_log.append(out.diagnostics)

    return SimulationLog(
        t=t_log,
        v_ref=v_ref_log,
        v_meas=v_meas_log,
        v_true=v_true_log,
        s=s_log,
        u=u_log,
        compute_us=compute_us_log,
        diagnostics=diag_log,
    )
