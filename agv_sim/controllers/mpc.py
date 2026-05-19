"""Traditional Model Predictive Controller (longitudinal).

Linear, time-invariant MPC with state x = [s, v] and SPLIT control input:
    u_throttle in [0, 1]   (drive force = b_drive * u_throttle)
    u_brake    in [0, 1]   (brake force = b_brake * u_brake)
Applied control to the plant: u = u_throttle - u_brake, in [-1, 1].

Splitting the control lets the MPC model the real asymmetry between the
plant's throttle force (F_drive_max = 60 kN) and brake force
(F_brake_max = 80 kN). Without the split, a single shared gain biases
closed-loop behaviour either in the throttle or the brake direction.

The two variables are non-negative and convex; their sum-of-squares cost
naturally drives the optimum to have at most one active at a time (using
both simultaneously is wasteful), so we do not need to enforce a
non-convex complementarity constraint.

Plant model used inside the QP (nominal, fixed):
    s_{k+1} = s_k + dt * v_k
    v_{k+1} = v_k + dt * (b_drive * u_thr_k - b_brake * u_brk_k - c_drag)
where
    b_drive = F_drive_max / m_nom      [m/s^2 per unit u_throttle]
    b_brake = F_brake_max / m_nom      [m/s^2 per unit u_brake]
    c_drag  = Cr_nom * g                [m/s^2]

This model still ignores motor lag and surface variation -- structural
slack that Intelligent MPC (Phase 4) will close with a neural residual.

Cost (per call, summed over the horizon):
    sum_{k=0}^{N-1} [ Q_v   * (v_k - v_ref_k)^2
                    + R_u   * (u_thr_k^2 + u_brk_k^2)
                    + R_du  * (u_k - u_{k-1})^2 ]
    + Q_term_s * (s_N - s_target)^2   (active iff Q_term_s > 0 at build)
    + Q_term_v *  v_N^2                (encourages stopping cleanly)

Constraints (hard):
    0 <= v_k <= v_max_k               (built from scenario preview)
    0 <= u_thr_k, u_brk_k <= 1
    |u_k - u_{k-1}| <= du_max

OSQP via CVXPY; DPP-compliant so the factorisation is reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cvxpy as cp
import numpy as np

from .base import Controller, ControlOutput


@dataclass
class MPCParameters:
    horizon: int = 20
    Q_v: float = 10.0
    R_u: float = 0.05            # control-effort weight (per direction)
    R_du: float = 1.0            # control-rate weight
    Q_term_s: float = 0.0        # terminal position weight (per-scenario override) (active iff > 0 at build)
    Q_term_v: float = 0.0        # terminal velocity-zero weight
    du_max: float = 0.3


@dataclass
class MPCPlantNominal:
    """Nominal plant parameters used inside the MPC's internal model."""

    m_nom: float = 14_000.0          # tractor + 6 t pallet
    cr_nom: float = 0.005            # rolling resistance
    g: float = 9.81
    f_drive_max: float = 60_000.0    # N - matches plant
    f_brake_max: float = 80_000.0    # N - matches plant

    @property
    def b_drive(self) -> float:
        return self.f_drive_max / self.m_nom

    @property
    def b_brake(self) -> float:
        return self.f_brake_max / self.m_nom

    @property
    def c_drag(self) -> float:
        return self.cr_nom * self.g


class MPCController(Controller):
    """Receding-horizon linear MPC with split throttle/brake. Solver: OSQP."""

    name = "mpc"

    def __init__(
        self,
        mpc_params: Optional[MPCParameters] = None,
        plant_nominal: Optional[MPCPlantNominal] = None,
        controller_dt: float = 0.05,
    ) -> None:
        self.mpc = mpc_params if mpc_params is not None else MPCParameters()
        self.plant = plant_nominal if plant_nominal is not None else MPCPlantNominal()
        self.mpc_dt = controller_dt
        self._u_prev = 0.0
        self._build_problem()

    def _build_problem(self) -> None:
        N = self.mpc.horizon
        dt = float(self.mpc_dt)
        b_dr = float(self.plant.b_drive)
        b_br = float(self.plant.b_brake)
        c = float(self.plant.c_drag)
        Q_term_s_const = float(self.mpc.Q_term_s)
        Q_term_v_const = float(self.mpc.Q_term_v)
        self._has_terminal_s = Q_term_s_const > 0.0

        # --- Decision variables ---
        self._s = cp.Variable(N + 1, name="s")
        self._v = cp.Variable(N + 1, name="v")
        self._u_thr = cp.Variable(N, name="u_thr", nonneg=True)
        self._u_brk = cp.Variable(N, name="u_brk", nonneg=True)
        # Slack on v_max to keep the QP feasible when the brake cannot meet the
        # constraint instantly (transients on entering a speed-limit zone).
        self._v_slack = cp.Variable(N + 1, name="v_slack", nonneg=True)

        # --- Parameters ---
        s0 = cp.Parameter(name="s0")
        v0 = cp.Parameter(name="v0")
        u_prev = cp.Parameter(name="u_prev")
        v_ref = cp.Parameter(N, name="v_ref")
        v_max = cp.Parameter(N + 1, nonneg=True, name="v_max")
        s_target = cp.Parameter(name="s_target")

        u_applied = self._u_thr - self._u_brk  # length-N affine expression

        # --- Cost ---
        cost = 0
        for k in range(N):
            cost = cost + self.mpc.Q_v * cp.square(self._v[k] - v_ref[k])
            cost = cost + self.mpc.R_u * (cp.square(self._u_thr[k]) + cp.square(self._u_brk[k]))
            if k == 0:
                cost = cost + self.mpc.R_du * cp.square(u_applied[0] - u_prev)
            else:
                cost = cost + self.mpc.R_du * cp.square(u_applied[k] - u_applied[k - 1])
        if self._has_terminal_s:
            cost = cost + Q_term_s_const * cp.square(self._s[N] - s_target)
        if Q_term_v_const > 0.0:
            cost = cost + Q_term_v_const * cp.square(self._v[N])
        # Slack penalty: large enough to prefer constraint satisfaction whenever possible.
        SLACK_PENALTY = 1e8     # extremely high so slack acts as near-hard constraint
        cost = cost + SLACK_PENALTY * cp.sum(cp.square(self._v_slack))

        # --- Constraints ---
        cons = [self._s[0] == s0, self._v[0] == v0]
        for k in range(N):
            cons.append(self._s[k + 1] == self._s[k] + dt * self._v[k])
            cons.append(
                self._v[k + 1]
                == self._v[k] + dt * (b_dr * self._u_thr[k] - b_br * self._u_brk[k] - c)
            )
            cons.append(self._u_thr[k] <= 1.0)
            cons.append(self._u_brk[k] <= 1.0)
            if k == 0:
                cons.append(u_applied[0] - u_prev <= self.mpc.du_max)
                cons.append(u_applied[0] - u_prev >= -self.mpc.du_max)
            else:
                cons.append(u_applied[k] - u_applied[k - 1] <= self.mpc.du_max)
                cons.append(u_applied[k] - u_applied[k - 1] >= -self.mpc.du_max)
        for k in range(N + 1):
            cons.append(self._v[k] >= 0.0)
            cons.append(self._v[k] <= v_max[k] + self._v_slack[k])

        self._problem = cp.Problem(cp.Minimize(cost), cons)
        self._params = {
            "s0": s0, "v0": v0, "u_prev": u_prev,
            "v_ref": v_ref, "v_max": v_max, "s_target": s_target,
        }

    def reset(self) -> None:
        self._u_prev = 0.0

    def step(
        self,
        v_meas: float,
        v_ref_now: float,
        dt: float,
        preview: Optional[Any] = None,
        s_meas: float = 0.0,
    ) -> ControlOutput:
        if preview is None or "v_ref" not in preview:
            raise ValueError("MPCController.step requires preview['v_ref']")
        if abs(dt - self.mpc_dt) > 1e-6:
            raise ValueError(f"MPC compiled for dt={self.mpc_dt}, called with dt={dt}")

        N = self.mpc.horizon
        v_ref_seq = np.asarray(preview["v_ref"], dtype=float)
        v_max_seq = np.asarray(preview.get("v_max", np.full(N + 1, 1e3)), dtype=float)
        if v_ref_seq.shape[0] != N:
            raise ValueError(f"preview['v_ref'] must have length {N}")
        if v_max_seq.shape[0] >= N + 1:
            v_max_seq = v_max_seq[: N + 1]
        else:
            v_max_seq = np.pad(v_max_seq, (0, N + 1 - v_max_seq.shape[0]), mode="edge")
        v_max_seq = np.maximum(v_max_seq, 0.0)
        # If v_meas exceeds v_max[0], bump it slightly so the initial-state
        # constraint stays feasible. The optional slack variables handle
        # deeper infeasibility along the horizon.
        if v_meas > v_max_seq[0]:
            v_max_seq[0] = v_meas + 1e-3

        s_target_val = 0.0
        if "s_target" in preview and preview["s_target"] is not None:
            s_target_val = float(np.asarray(preview["s_target"]).item())

        self._params["s0"].value = s_meas
        self._params["v0"].value = v_meas
        self._params["u_prev"].value = self._u_prev
        self._params["v_ref"].value = v_ref_seq
        self._params["v_max"].value = v_max_seq
        self._params["s_target"].value = s_target_val

        try:
            self._problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            self._u_prev = 0.0
            return ControlOutput(u=0.0, diagnostics={"error": str(e), "status": "exception"})

        status = self._problem.status
        if (self._u_thr.value is None or self._u_brk.value is None
                or status not in ("optimal", "optimal_inaccurate")):
            u = 0.0
            v_plan = np.full(N + 1, np.nan)
            s_plan = np.full(N + 1, np.nan)
        else:
            u_thr0 = float(np.clip(self._u_thr.value[0], 0.0, 1.0))
            u_brk0 = float(np.clip(self._u_brk.value[0], 0.0, 1.0))
            u = float(np.clip(u_thr0 - u_brk0, -1.0, 1.0))
            v_plan = np.asarray(self._v.value).copy()
            s_plan = np.asarray(self._s.value).copy()

        self._u_prev = u
        return ControlOutput(
            u=u,
            diagnostics={
                "status": status,
                "v_ref0": v_ref_seq[0],
                "v_max0": v_max_seq[0],
                "v_plan": v_plan,
                "s_plan": s_plan,
                "s_target": s_target_val,
            },
        )
