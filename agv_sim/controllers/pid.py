"""Traditional PID controller.

Discrete-time PID with:
  * derivative filtering (first-order low-pass on the derivative term)
  * back-calculation anti-windup on the integral term
  * symmetric output clamping to the actuator range
  * velocity-form computation kept stateful (position-form internally; we
    expose the standard parallel-form parameters).

This is the baseline controller against which the other three are judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .base import Controller, ControlOutput


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    n_d: float = 20.0       # derivative filter bandwidth [rad/s]; lower = smoother
    u_min: float = -1.0
    u_max: float = 1.0
    kt: float = 1.0         # back-calculation anti-windup gain


class PIDController(Controller):
    """Parallel-form PID with anti-windup and filtered derivative."""

    name = "pid"

    def __init__(self, gains: PIDGains) -> None:
        self.g = gains
        self._integral = 0.0
        self._d_filtered = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._d_filtered = 0.0
        self._prev_error = None

    def step(
        self,
        v_meas: float,
        v_ref: float,
        dt: float,
        preview: Optional[Any] = None,
        s_meas: float = 0.0,
    ) -> ControlOutput:
        error = v_ref - v_meas

        # Proportional.
        p_term = self.g.kp * error

        # Derivative on error, first-order filtered.
        if self._prev_error is None:
            de = 0.0
        else:
            de = (error - self._prev_error) / dt
        alpha = self.g.n_d * dt / (1.0 + self.g.n_d * dt)
        self._d_filtered += alpha * (self.g.kd * de - self._d_filtered)
        d_term = self._d_filtered

        # Provisional integral.
        self._integral += self.g.ki * error * dt
        i_term = self._integral

        u_unsat = p_term + i_term + d_term

        # Saturate.
        if u_unsat > self.g.u_max:
            u = self.g.u_max
        elif u_unsat < self.g.u_min:
            u = self.g.u_min
        else:
            u = u_unsat

        # Back-calculation anti-windup.
        if u != u_unsat:
            self._integral += self.g.kt * (u - u_unsat) * dt

        self._prev_error = error

        return ControlOutput(
            u=float(u),
            diagnostics={
                "error": error,
                "p": p_term,
                "i": i_term,
                "d": d_term,
                "u_unsat": u_unsat,
                "kp": self.g.kp,
                "ki": self.g.ki,
                "kd": self.g.kd,
            },
        )
