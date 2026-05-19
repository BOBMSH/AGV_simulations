"""Scenario A — Smooth concrete cruise.

Designed to favour Traditional PID. Justification:

* Surface: uniform smooth concrete (Cr = 0.004), no variation.
* Payload: constant 4 t cargo (well within nominal).
* Grade: flat (theta = 0).
* No binding constraints, no preview value, no parameter drift.
* Reference: a benign trapezoidal velocity profile typical of a warehouse
  aisle run — accelerate from 0 to 1.5 m/s (~5 km/h), cruise, decelerate to
  stop. Accel/decel within the comfort limit so motor saturation is inactive.

Under these conditions the plant is effectively LTI in a quasi-static sense.
A well-tuned PID achieves tracking performance indistinguishable from MPC or
IMPC, at a fraction of the per-tick compute. This scenario establishes the
"Occam's razor" case: intelligent control is not always justified.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..plant import EnvironmentProfile
from .base import Scenario, ScenarioConstraints


@dataclass
class TrapezoidalProfile:
    """A 4-phase trapezoidal velocity profile: ramp up, cruise, ramp down, hold zero."""

    v_cruise: float = 1.5       # m/s ≈ 5.4 km/h, indoor AGV speed
    t_accel: float = 4.0        # ramp-up duration [s]
    t_cruise: float = 18.0      # cruise duration [s]
    t_decel: float = 4.0        # ramp-down duration [s]
    t_pre_start: float = 1.0    # zero-hold at start
    t_tail: float = 3.0         # zero-hold at end

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


class ScenarioA(Scenario):
    """Smooth concrete cruise — favours Traditional PID."""

    name = "A"
    description = "Smooth concrete warehouse aisle cruise (favours Traditional PID)"
    favored_controller = "pid"

    def __init__(self) -> None:
        self._profile = TrapezoidalProfile()
        self.duration = self._profile.total_duration

    def reference_velocity(self, t: float) -> float:
        return self._profile.value(t)

    def environment(self) -> EnvironmentProfile:
        return EnvironmentProfile(
            surface_cr=lambda s, t: 0.004,   # smooth concrete
            grade_rad=lambda s, t: 0.0,
            payload_kg=lambda s, t: 4000.0,  # 4 t cargo, constant
            include_aero=False,              # indoor speeds
        )

    def constraints(self) -> ScenarioConstraints:
        return ScenarioConstraints(
            v_min=0.0,
            v_max=2.5,      # not binding given v_cruise=1.5
            a_max=1.0,      # not binding given the gentle ramp
        )
