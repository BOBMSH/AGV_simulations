"""Intelligent PID controller (NN gain scheduler).

Same parallel-form PID loop as Traditional PID, but with the gains
(Kp, Ki, Kd) re-computed each control tick by a small MLP whose input
is the AGV's current operating-point estimate (rolling-resistance
coefficient, payload mass). The MLP is trained offline by training.py.

Operating-point estimates come via the scenario's preview dict:
    preview["surface_cr"]  : current rolling-resistance estimate
    preview["payload_kg"]  : current payload estimate

If these are missing the controller falls back to the baseline gains
supplied at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import Controller, ControlOutput
from .pid import PIDGains
from ..nn import MLP


@dataclass
class IPIDConfig:
    """Configuration for the IPID controller."""

    model_path: Path
    fallback_gains: PIDGains = None  # used when preview lacks operating-point info
    smoothing_alpha: float = 0.4     # 1 = no smoothing, 0 = freeze gains
    n_d: float = 15.0                # derivative filter bandwidth
    u_min: float = -1.0
    u_max: float = 1.0
    kt: float = 1.0                  # anti-windup gain


class IntelligentPIDController(Controller):
    """PID with NN-driven gain scheduling."""

    name = "ipid"

    def __init__(self, cfg: IPIDConfig) -> None:
        self.cfg = cfg
        if cfg.fallback_gains is None:
            self.cfg.fallback_gains = PIDGains(kp=0.85, ki=0.45, kd=0.05)
        self._mlp = MLP.load(Path(cfg.model_path))
        # Internal PID state
        self._integral = 0.0
        self._d_filtered = 0.0
        self._prev_error: Optional[float] = None
        # Smoothed gains (avoid step changes when MLP output jitters)
        self._kp = self.cfg.fallback_gains.kp
        self._ki = self.cfg.fallback_gains.ki
        self._kd = self.cfg.fallback_gains.kd

    def reset(self) -> None:
        self._integral = 0.0
        self._d_filtered = 0.0
        self._prev_error = None
        self._kp = self.cfg.fallback_gains.kp
        self._ki = self.cfg.fallback_gains.ki
        self._kd = self.cfg.fallback_gains.kd

    # ------------------------------------------------------------------
    # NN inference
    # ------------------------------------------------------------------

    def _infer_gains(self, preview) -> Optional[np.ndarray]:
        if preview is None:
            return None
        try:
            cr = float(np.asarray(preview["surface_cr"]).item())
            payload = float(np.asarray(preview["payload_kg"]).item())
        except (KeyError, TypeError, ValueError):
            return None
        x = np.array([cr, payload], dtype=np.float64)
        return self._mlp.forward(x)  # (3,) array

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------

    def step(
        self,
        v_meas: float,
        v_ref: float,
        dt: float,
        preview: Optional[Any] = None,
        s_meas: float = 0.0,
    ) -> ControlOutput:
        # --- Update gains via NN inference --------------------------------
        gains = self._infer_gains(preview)
        if gains is not None:
            kp_t, ki_t, kd_t = float(gains[0]), float(gains[1]), float(gains[2])
        else:
            kp_t = self.cfg.fallback_gains.kp
            ki_t = self.cfg.fallback_gains.ki
            kd_t = self.cfg.fallback_gains.kd
        # Smooth so the gains don't change abruptly at surface boundaries.
        a = self.cfg.smoothing_alpha
        self._kp = a * kp_t + (1.0 - a) * self._kp
        self._ki = a * ki_t + (1.0 - a) * self._ki
        self._kd = a * kd_t + (1.0 - a) * self._kd

        # --- Standard PID with current gains ------------------------------
        error = v_ref - v_meas
        p_term = self._kp * error

        if self._prev_error is None:
            de = 0.0
        else:
            de = (error - self._prev_error) / dt
        alpha = self.cfg.n_d * dt / (1.0 + self.cfg.n_d * dt)
        self._d_filtered += alpha * (self._kd * de - self._d_filtered)
        d_term = self._d_filtered

        self._integral += self._ki * error * dt
        i_term = self._integral

        u_unsat = p_term + i_term + d_term
        if u_unsat > self.cfg.u_max:
            u = self.cfg.u_max
        elif u_unsat < self.cfg.u_min:
            u = self.cfg.u_min
        else:
            u = u_unsat
        if u != u_unsat:
            self._integral += self.cfg.kt * (u - u_unsat) * dt

        self._prev_error = error

        return ControlOutput(
            u=float(u),
            diagnostics={
                "error": error,
                "kp": self._kp,
                "ki": self._ki,
                "kd": self._kd,
                "kp_nn": kp_t,
                "ki_nn": ki_t,
                "kd_nn": kd_t,
                "p": p_term, "i": i_term, "d": d_term,
                "u_unsat": u_unsat,
            },
        )
