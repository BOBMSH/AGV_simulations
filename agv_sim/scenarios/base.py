"""Scenario abstract base class.

A scenario fully specifies a benchmark run by providing:
  * a reference velocity profile v_ref(t)
  * an environment profile (surface mu, grade, payload) consumed by the plant
  * any constraints (v_max(s), a_max, position-stop targets) that controllers
    should honour
  * scenario metadata (name, duration, narrative justification)

The preview() method packages all forward-looking information into a dict so
that MPC-class controllers can consume v_ref / v_max / s_target uniformly,
while PID-class controllers can ignore everything except v_ref.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..plant import EnvironmentProfile


@dataclass
class ScenarioConstraints:
    """Optional hard constraints. MPC consumes them; PID/IPID ignore."""

    v_min: float = 0.0
    v_max: float = 28.0
    a_max: float = 2.5
    u_max: float = 1.0
    u_min: float = -1.0
    du_max: float = 0.4
    s_stop: Optional[float] = None
    s_stop_tolerance: float = 0.05


class Scenario(ABC):
    """Abstract benchmark scenario."""

    name: str = "base"
    description: str = ""
    favored_controller: str = ""
    duration: float = 60.0

    @abstractmethod
    def reference_velocity(self, t: float) -> float:
        """Return the reference velocity at time t [m/s]."""

    @abstractmethod
    def environment(self) -> EnvironmentProfile:
        """Return the environment profile to hand to the plant."""

    def constraints(self) -> ScenarioConstraints:
        """Default constraints; override per scenario."""
        return ScenarioConstraints()

    def speed_limit_at(self, s: float) -> float:
        """v_max as a function of position. Default = global cap."""
        return self.constraints().v_max

    def preview(
        self,
        t: float,
        horizon: int,
        dt: float,
        s_now: float = 0.0,
        v_now: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Return forward-looking trajectories over the next `horizon` ticks.

        Default v_max prediction assumes zero-acceleration forecast from
        (s_now, v_now); first-order accurate, which is fine because the MPC
        re-builds this every tick.
        """
        v_ref = np.array([self.reference_velocity(t + k * dt) for k in range(horizon)])
        s_pred = s_now + v_now * np.arange(horizon + 1) * dt
        v_max = np.array([self.speed_limit_at(float(s)) for s in s_pred])
        cons = self.constraints()
        out: Dict[str, np.ndarray] = {
            "v_ref": v_ref,
            "v_max": v_max,
            "s_pred": s_pred,
        }
        if cons.s_stop is not None:
            out["s_target"] = np.array([cons.s_stop], dtype=float)
        return out
