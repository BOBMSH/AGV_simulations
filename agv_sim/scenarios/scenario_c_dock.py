"""Scenario C - Tight dock approach.

Designed to favour Traditional MPC. Justification:

Two complementary structural advantages for MPC in this scenario:

  (1) **Safety envelope compliance.** A piecewise-linear speed limit v_max(s)
      mimics a warehouse SICK-scanner hierarchy: full 1.5 m/s outside the
      funnel (s < 35 m), linearly decreasing as the AGV approaches the
      dock, with a small positive floor near the dock face so the AGV can
      crawl the last metres. PID has no way to anticipate the upcoming
      envelope drop; lag during the cruise -> funnel transition causes it
      to violate the limit. MPC's 1 s preview bakes v_max(s) into its QP
      and brakes pre-emptively.

  (2) **Stopping accuracy.** v_ref(t) is calibrated so that, if tracked
      perfectly, integration of the trapezoid yields s = s_dock at v = 0.
      PID's velocity lag during the deceleration phase causes the
      integrated position to overshoot the dock face. MPC tracks v_ref
      more tightly because it sees the upcoming v_ref drops in its
      horizon, and lands within the +/-5 cm tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..plant import EnvironmentProfile
from .base import Scenario, ScenarioConstraints


@dataclass
class DockProfile:
    """Trapezoidal velocity profile calibrated to stop at s_dock if tracked perfectly."""

    v_cruise: float = 1.5
    t_pre_start: float = 1.0
    t_accel: float = 3.0
    t_cruise: float = 25.22  # calibrated below
    t_decel: float = 10.0
    t_tail: float = 5.0
    s_dock: float = 50.0

    def __post_init__(self) -> None:
        s_accel = 0.5 * self.v_cruise * self.t_accel
        s_decel = 0.5 * self.v_cruise * self.t_decel
        target_cruise_dist = self.s_dock - s_accel - s_decel
        self.t_cruise = target_cruise_dist / self.v_cruise

    @property
    def total_duration(self) -> float:
        return (
            self.t_pre_start
            + self.t_accel
            + self.t_cruise
            + self.t_decel
            + self.t_tail
        )

    def value(self, t: float) -> float:
        t1 = self.t_pre_start
        t2 = t1 + self.t_accel
        t3 = t2 + self.t_cruise
        t4 = t3 + self.t_decel
        if t < t1:
            return 0.0
        elif t < t2:
            return self.v_cruise * (t - t1) / self.t_accel
        elif t < t3:
            return self.v_cruise
        elif t < t4:
            return self.v_cruise * (1.0 - (t - t3) / self.t_decel)
        else:
            return 0.0


class ScenarioC(Scenario):
    """Tight dock approach - favours Traditional MPC."""

    name = "C"
    description = "Tight dock with v_max(s) envelope & +/-5cm stop tolerance (favours Traditional MPC)"
    favored_controller = "mpc"

    # v_max(s) envelope: piecewise-linear "funnel".
    #   v_max(s) = max( v_floor , min( V_CAP , K_RAMP * (S_DOCK - s) ) )
    # Holds at V_CAP for s < S_FUNNEL_START, ramps linearly to v_floor at
    # the dock face. The positive floor (0.05 m/s) makes the envelope
    # *physically achievable while stopping* and avoids QP infeasibility
    # at the dock.
    V_CAP: float = 1.5
    K_RAMP: float = 0.7      # steep: 1.4 m/s drop per 2m, only active near dock
    S_DOCK: float = 50.0
    V_FLOOR: float = 0.05
    S_FUNNEL_START: float = 48.0  # funnel bites only in the last 2 m

    def __init__(self) -> None:
        self._profile = DockProfile(s_dock=self.S_DOCK, v_cruise=self.V_CAP)
        self.duration = self._profile.total_duration

    def reference_velocity(self, t: float) -> float:
        return self._profile.value(t)

    def environment(self) -> EnvironmentProfile:
        return EnvironmentProfile(
            surface_cr=lambda s, t: 0.005,
            grade_rad=lambda s, t: 0.0,
            payload_kg=lambda s, t: 6000.0,
            include_aero=False,
        )

    def speed_limit_at(self, s: float) -> float:
        # Outside the funnel: full cruise cap.
        if s < self.S_FUNNEL_START:
            return self.V_CAP
        # Inside the funnel: linear ramp down to V_FLOOR at the dock.
        envelope = self.K_RAMP * (self.S_DOCK - s)
        return max(self.V_FLOOR, min(self.V_CAP, envelope))

    def constraints(self) -> ScenarioConstraints:
        return ScenarioConstraints(
            v_min=0.0,
            v_max=self.V_CAP,
            a_max=1.5,
            u_max=1.0,
            u_min=-1.0,
            du_max=0.3,
            s_stop=self.S_DOCK,
            s_stop_tolerance=0.05,
        )
