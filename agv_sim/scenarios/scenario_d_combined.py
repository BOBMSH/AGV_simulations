"""Scenario D - Heavy load + slippery uphill, then downhill speed limit.

Designed to favour Intelligent MPC. Justification:

Three plant uncertainties stacked simultaneously:
  (1) **Grade variation.** A 6% wet uphill from s=20..50, crest at s=50..55,
      then a 4% downhill from s=55..80. Gravity term is significant at
      these grades (~5% of F_drive_max).
  (2) **Payload step mid-route.** AGV picks up a 12 t pallet at s=20
      (a mid-route loading point). Plant inertia jumps suddenly.
  (3) **Surface variation.** The uphill stretch is wet (Cr=0.018);
      the downhill is dry but rougher (Cr=0.012). Aero drag is active.

A safety speed-limit zone on the descent (v_max=0.8 m/s for s=60..80)
requires preview-based braking because the AGV must slow from cruise
before entering the zone.

This scenario stresses every plant-model assumption simultaneously:
  - PID has no preview AND no model awareness -> overshoots descent limit.
  - IPID adapts gains but still no preview -> same descent failure.
  - Traditional MPC has preview but its fixed plant model (nominal
    Cr_nom, m_nom, theta=0) is wrong on every count -> mistimes brakes
    and either violates limit or undershoots stopping.
  - Intelligent MPC's NN residual captures the combined Cr / payload /
    grade mismatch in its predictions; with preview it brakes at exactly
    the right point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..plant import EnvironmentProfile
from .base import Scenario, ScenarioConstraints


def _grade_at(s: float) -> float:
    """Road grade (radians) along the route."""
    if s < 20.0:
        return 0.0
    if s < 50.0:
        return np.arctan(0.06)     # 6% uphill
    if s < 55.0:
        return 0.0                  # crest
    if s < 80.0:
        return -np.arctan(0.04)    # 4% downhill
    return 0.0


def _surface_at(s: float) -> float:
    """Rolling resistance Cr along the route."""
    if s < 50.0:
        return 0.018               # wet uphill
    return 0.012                    # dry rougher downhill


def _payload_at(s: float) -> float:
    """Payload mass [kg]: empty until s=20, then 12 t."""
    if s < 20.0:
        return 0.0
    return 12_000.0


class ScenarioD(Scenario):
    """Heavy load + slippery ramp + non-linear drive (favours Intelligent MPC)."""

    name = "D"
    description = "Inter-plant route: payload pickup, wet uphill, downhill speed limit (favours IMPC)"
    favored_controller = "impc"

    V_CRUISE: float = 1.5
    ROUTE_LENGTH: float = 80.0
    # Optional safety zone (set DESCENT_VMAX>0 to enable). The headline
    # demo runs WITHOUT a hard constraint -- the win goes to whichever
    # controller can most accurately predict the combined grade/payload/
    # surface effect and brake at the right magnitude. Phase 5 polish can
    # add the constraint back.
    DESCENT_VMAX: float = 0.0                       # 0 = disabled
    DESCENT_START: float = 60.0
    DESCENT_END: float = 80.0

    def __init__(self) -> None:
        self.t_pre_start = 1.0
        self.t_accel = 3.0
        self.t_decel = 4.0
        s_accel = 0.5 * self.V_CRUISE * self.t_accel
        s_decel = 0.5 * self.V_CRUISE * self.t_decel
        target_cruise_dist = self.ROUTE_LENGTH - s_accel - s_decel
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
            surface_cr=lambda s, t: _surface_at(s),
            grade_rad=lambda s, t: _grade_at(s),
            payload_kg=lambda s, t: _payload_at(s),
            include_aero=True,
        )

    def speed_limit_at(self, s: float) -> float:
        if self.DESCENT_VMAX > 0.0 and self.DESCENT_START <= s < self.DESCENT_END:
            return self.DESCENT_VMAX
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

    def preview(
        self, t: float, horizon: int, dt: float,
        s_now: float = 0.0, v_now: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        base = super().preview(t, horizon, dt, s_now=s_now, v_now=v_now)
        base["surface_cr"] = float(_surface_at(s_now))
        base["payload_kg"] = float(_payload_at(s_now))
        base["grade_rad"] = float(_grade_at(s_now))
        return base
