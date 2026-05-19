"""
AGV longitudinal dynamics.

The model captures the essential physics needed to differentiate the four
controllers without dragging in a full vehicle dynamics engine:

    m(t) * dv/dt = F_drive - F_roll - F_grade - F_aero - F_brake

where
    m(t)         : payload-dependent mass [kg]
    F_drive      : drive force from the motor, first-order lag + saturation
    F_roll       : rolling resistance = Cr(s) * m * g * cos(theta(s))
    F_grade      : gravitational along-slope = m * g * sin(theta(s))
    F_aero       : 0.5 * rho * Cd * A * v^2 (negligible indoors)
    F_brake      : optional commanded braking force

Integration is RK4 at a configurable simulation timestep (default 100 Hz).
Controllers operate on a slower clock (default 20 Hz) and only see noisy
encoder measurements of velocity.

Scenario objects supply the time-varying parameters (payload mass, surface
friction coefficient, grade) as functions of either time or path coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class AGVParameters:
    """Static plant parameters shared across scenarios."""

    # --- vehicle ---
    mass_empty: float = 8000.0          # tractor curb mass [kg] (Actros L scale)
    mass_payload_max: float = 18000.0   # max payload [kg]
    # --- forces ---
    g: float = 9.81                     # gravity [m/s^2]
    rho_air: float = 1.225              # air density [kg/m^3]
    cd: float = 0.6                     # drag coefficient
    frontal_area: float = 10.0          # m^2 (Actros L cabin)
    # --- actuation ---
    f_drive_max: float = 60_000.0       # max drive force [N]
    f_brake_max: float = 80_000.0       # max brake force [N]
    motor_tau: float = 0.25             # motor first-order lag time constant [s]
    # --- limits ---
    v_max: float = 28.0                 # 100 km/h hard cap
    a_max: float = 2.5                  # comfort accel limit [m/s^2]
    # --- sensor ---
    encoder_noise_std: float = 0.02     # m/s std dev (≈ 0.07 km/h)
    encoder_bias: float = 0.0           # m/s persistent bias
    # --- numerical ---
    dt_sim: float = 0.01                # 100 Hz physics
    dt_ctrl: float = 0.05               # 20 Hz controller


@dataclass
class AGVState:
    """Time-varying state of the plant."""

    s: float = 0.0          # path coordinate [m]
    v: float = 0.0          # velocity [m/s]
    f_drive: float = 0.0    # current drive force after motor lag [N]
    t: float = 0.0          # simulation time [s]


# ---------------------------------------------------------------------------
# Scenario hooks — these are simple callables returning a float given (s, t).
# Scenarios pass concrete implementations in. The defaults give a benign,
# constant-condition environment.
# ---------------------------------------------------------------------------

ProfileFn = Callable[[float, float], float]  # (s, t) -> value


def _const(value: float) -> ProfileFn:
    """Return a profile function that ignores s and t."""
    return lambda s, t: value


@dataclass
class EnvironmentProfile:
    """Time- and position-varying environment parameters.

    Each field is a callable (s, t) -> float so scenarios can express both
    space-dependent (surface μ along the aisle) and time-dependent (mid-route
    payload step) variations cleanly.
    """

    surface_cr: ProfileFn = field(default_factory=lambda: _const(0.006))
    grade_rad: ProfileFn = field(default_factory=lambda: _const(0.0))
    payload_kg: ProfileFn = field(default_factory=lambda: _const(0.0))
    include_aero: bool = False


class AGVPlant:
    """Longitudinal AGV plant.

    The plant exposes:
        - step(u, dt): advance the state by one physics step under control u
        - measure(): return a noisy encoder reading of velocity
        - reset(): reset to s=0, v=0, f_drive=0, t=0

    Control input u ∈ [-1, 1]:
        u > 0 commands a fraction u of f_drive_max as forward thrust
        u < 0 commands a fraction |u| of f_brake_max as brake
    The motor first-order lag is applied to the resulting commanded force.
    """

    def __init__(
        self,
        params: AGVParameters,
        env: EnvironmentProfile,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.p = params
        self.env = env
        self.state = AGVState()
        self.rng = rng if rng is not None else np.random.default_rng(0)

    # ---- public API -------------------------------------------------------

    def reset(self, v0: float = 0.0, s0: float = 0.0) -> None:
        self.state = AGVState(s=s0, v=v0, f_drive=0.0, t=0.0)

    def step(self, u: float, dt: float) -> AGVState:
        """Advance one RK4 step of length dt under control input u in [-1, 1]."""
        u = float(np.clip(u, -1.0, 1.0))
        # Commanded force after the saturating actuator map.
        if u >= 0.0:
            f_cmd = u * self.p.f_drive_max
        else:
            f_cmd = u * self.p.f_brake_max  # negative

        # RK4 on state vector x = [s, v, f_drive].
        x = np.array([self.state.s, self.state.v, self.state.f_drive])
        t0 = self.state.t

        k1 = self._dxdt(x, t0, f_cmd)
        k2 = self._dxdt(x + 0.5 * dt * k1, t0 + 0.5 * dt, f_cmd)
        k3 = self._dxdt(x + 0.5 * dt * k2, t0 + 0.5 * dt, f_cmd)
        k4 = self._dxdt(x + dt * k3, t0 + dt, f_cmd)
        x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Clamp velocity to non-negative (AGV cannot reverse in our scenarios
        # unless explicitly modeled later) and below v_max.
        x_new[1] = float(np.clip(x_new[1], 0.0, self.p.v_max))

        self.state = AGVState(
            s=float(x_new[0]),
            v=float(x_new[1]),
            f_drive=float(x_new[2]),
            t=t0 + dt,
        )
        return self.state

    def measure(self) -> float:
        """Return noisy encoder reading of velocity."""
        return self.state.v + self.p.encoder_bias + self.rng.normal(0.0, self.p.encoder_noise_std)

    def total_mass(self, s: float, t: float) -> float:
        return self.p.mass_empty + self.env.payload_kg(s, t)

    # ---- internals --------------------------------------------------------

    def _dxdt(self, x: np.ndarray, t: float, f_cmd: float) -> np.ndarray:
        s, v, f_drive = x

        # Time-varying parameters.
        m = self.total_mass(s, t)
        cr = self.env.surface_cr(s, t)
        theta = self.env.grade_rad(s, t)
        g = self.p.g

        # Forces.
        f_roll = cr * m * g * np.cos(theta) * np.sign(max(v, 1e-3))
        f_grade = m * g * np.sin(theta)
        if self.env.include_aero:
            f_aero = 0.5 * self.p.rho_air * self.p.cd * self.p.frontal_area * v * v * np.sign(v)
        else:
            f_aero = 0.0

        # Motor first-order lag: tau * d f_drive / dt = f_cmd - f_drive
        df_drive = (f_cmd - f_drive) / self.p.motor_tau

        # Net force.
        f_net = f_drive - f_roll - f_grade - f_aero

        dv = f_net / m
        ds = v

        return np.array([ds, dv, df_drive])
