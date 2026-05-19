"""Scenario B - Surface-transition loading run.

Designed to favour Intelligent PID. Justification:

The AGV travels along a long aisle whose floor surface changes in four
segments, mimicking a realistic warehouse route from the parking bay
to a wet-floor area to the loading mat. Each surface has a different
rolling-resistance coefficient Cr, which acts as a multiplicative
disturbance on the plant. There are no constraints binding, no preview
value, no payload variation.

  s in [0, 20)  : smooth concrete  Cr=0.004
  s in [20, 40) : painted line     Cr=0.008
  s in [40, 60) : wet patch        Cr=0.014
  s in [60, 80) : rubber mat       Cr=0.020

Reference: trapezoidal up to cruise at 1.5 m/s, cruise, ramp down.
Payload: constant 6000 kg.
Grade: flat.

Failure mode of fixed-gain PID: gains tuned for one surface (typically
the median) detune on the extremes. On smooth concrete the controller
is overly aggressive (overshoot, oscillation). On the wet/mat sections
it is under-damped relative to the increased drag and lags badly.

Intelligent PID receives the current surface Cr via the scenario's
preview (a virtual sensor proxy for vision/RFID floor markers in a
real warehouse) and re-tunes Kp/Ki/Kd on the fly via its NN gain
scheduler. Smooth gain transitions are achieved by the controller's
internal exponential smoothing.

MPC, with its fixed nominal plant model, also degrades on the extreme
surfaces (its model of c_drag is fixed); only the IMPC of Phase 4
will properly adapt the internal plant model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..plant import EnvironmentProfile
from .base import Scenario, ScenarioConstraints


@dataclass
class _SurfaceSegment:
    s_start: float
    s_end: float
    cr: float
    label: str


# Four surface segments along an 80 m aisle.
SURFACES: List[_SurfaceSegment] = [
    _SurfaceSegment( 0.0, 20.0, 0.004, "smooth concrete"),
    _SurfaceSegment(20.0, 40.0, 0.012, "painted/scuffed"),
    _SurfaceSegment(40.0, 60.0, 0.022, "wet floor"),
    _SurfaceSegment(60.0, 80.0, 0.030, "rubber loading mat"),
]


def surface_cr_at(s: float) -> float:
    """Return rolling-resistance coefficient at position s."""
    if s < SURFACES[0].s_start:
        return SURFACES[0].cr
    for seg in SURFACES:
        if seg.s_start <= s < seg.s_end:
            return seg.cr
    return SURFACES[-1].cr




def payload_at(s: float) -> float:
    """Cargo payload along the aisle (loading pickup at zone boundaries)."""
    if s < 20.0:
        return 0.0       # empty
    if s < 40.0:
        return 4_000.0   # picked up first pallet
    if s < 60.0:
        return 9_000.0   # second pallet
    return 14_000.0      # full load


class ScenarioB(Scenario):
    """Surface-transition loading run - favours Intelligent PID."""

    name = "B"
    description = (
        "80 m aisle: 4 surfaces + 4 payload pickup stations "
        "(favours Intelligent PID)"
    )
    favored_controller = "ipid"

    V_CRUISE: float = 1.5
    # Payload grows along the path (cargo loading stations).
    AISLE_LENGTH: float = 80.0

    def __init__(self) -> None:
        # Trapezoidal profile calibrated to cover the aisle.
        self.t_pre_start = 1.0
        self.t_accel = 3.0
        self.t_decel = 4.0
        s_accel = 0.5 * self.V_CRUISE * self.t_accel
        s_decel = 0.5 * self.V_CRUISE * self.t_decel
        target_cruise_dist = self.AISLE_LENGTH - s_accel - s_decel
        self.t_cruise = target_cruise_dist / self.V_CRUISE
        self.t_tail = 3.0
        self.duration = (
            self.t_pre_start + self.t_accel + self.t_cruise + self.t_decel + self.t_tail
        )

    def reference_velocity(self, t: float) -> float:
        t1 = self.t_pre_start
        t2 = t1 + self.t_accel
        t3 = t2 + self.t_cruise
        t4 = t3 + self.t_decel
        if t < t1:
            return 0.0
        if t < t2:
            return self.V_CRUISE * (t - t1) / self.t_accel
        if t < t3:
            return self.V_CRUISE
        if t < t4:
            return self.V_CRUISE * (1.0 - (t - t3) / self.t_decel)
        return 0.0

    def environment(self) -> EnvironmentProfile:
        return EnvironmentProfile(
            surface_cr=lambda s, t: surface_cr_at(s),
            grade_rad=lambda s, t: 0.0,
            payload_kg=lambda s, t: payload_at(s),
            include_aero=False,
        )

    def speed_limit_at(self, s: float) -> float:
        # No funnel — only a global cap.
        return self.constraints().v_max

    def constraints(self) -> ScenarioConstraints:
        return ScenarioConstraints(
            v_min=0.0,
            v_max=2.0,
            a_max=2.0,
            u_max=1.0,
            u_min=-1.0,
            du_max=0.4,
        )

    # --- Augmented preview ----------------------------------------------------

    def preview(
        self,
        t: float,
        horizon: int,
        dt: float,
        s_now: float = 0.0,
        v_now: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Adds 'surface_cr' and 'payload_kg' for the IPID's NN input."""
        base = super().preview(t, horizon, dt, s_now=s_now, v_now=v_now)
        base["surface_cr"] = float(surface_cr_at(s_now))
        base["payload_kg"] = float(payload_at(s_now))
        return base
