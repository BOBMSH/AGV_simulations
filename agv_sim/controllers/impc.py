"""Intelligent Model Predictive Controller (NN-augmented plant model).

Builds on the Traditional MPC by adding a neural-network residual to the
internal plant model. At each control tick the residual is *linearised*
around the current operating point and folded into the QP's dynamics
constraints as a small SQP step:

    v_{k+1} = v_k + dt * (b_dr * u_thr_k - b_br * u_brk_k - c_drag)
              + delta_v(v_k, u_thr_k, u_brk_k, cr, payload, grade)

After linearisation around (v_0, u_thr_0, u_brk_0):
    delta_v ~= offset + J_v * v_k + J_thr * u_thr_k + J_brk * u_brk_k

Substituting gives an AFFINE dynamics constraint:
    v_{k+1} = (1 + J_v) * v_k
              + (dt * b_dr + J_thr) * u_thr_k
              + (-dt * b_br + J_brk) * u_brk_k
              + (offset - dt * c_drag)

The coefficients on the LINEAR terms are cvxpy Parameters; the QP
structure is preserved, so OSQP reuses its factorisation across solves.
The same linearisation is applied at all horizon steps -- a single SQP
step around the current state. With horizon 20 ticks (1 s) this is
accurate enough for our scenarios and keeps the per-tick solve <10 ms.

Environment inputs (cr, payload, grade) are pulled from the scenario
preview and treated as fixed over the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cvxpy as cp
import numpy as np

from ..nn import MLP
from .base import Controller, ControlOutput
from .mpc import MPCParameters, MPCPlantNominal


@dataclass
class IMPCConfig:
    """Configuration for the IMPC controller."""

    residual_model_path: Path
    mpc_params: MPCParameters = None
    plant_nominal: MPCPlantNominal = None
    controller_dt: float = 0.05
    # Damp the influence of the NN residual to be safe with model uncertainty.
    residual_scale: float = 1.0


class IntelligentMPCController(Controller):
    """Receding-horizon MPC with NN-residual plant model."""

    name = "impc"

    def __init__(self, cfg: IMPCConfig) -> None:
        if cfg.mpc_params is None:
            cfg.mpc_params = MPCParameters()
        if cfg.plant_nominal is None:
            cfg.plant_nominal = MPCPlantNominal()
        self.cfg = cfg
        self.mpc = cfg.mpc_params
        self.plant = cfg.plant_nominal
        self.mpc_dt = cfg.controller_dt
        self._mlp = MLP.load(Path(cfg.residual_model_path))
        self._u_thr_prev = 0.0
        self._u_brk_prev = 0.0
        self._build_problem()

    # ------------------------------------------------------------------

    def _build_problem(self) -> None:
        N = self.mpc.horizon
        dt = float(self.mpc_dt)
        b_dr = float(self.plant.b_drive)
        b_br = float(self.plant.b_brake)
        c = float(self.plant.c_drag)
        Q_term_s_const = float(self.mpc.Q_term_s)
        Q_term_v_const = float(self.mpc.Q_term_v)
        self._has_terminal_s = Q_term_s_const > 0.0

        # Variables
        self._s = cp.Variable(N + 1, name="s")
        self._v = cp.Variable(N + 1, name="v")
        self._u_thr = cp.Variable(N, name="u_thr", nonneg=True)
        self._u_brk = cp.Variable(N, name="u_brk", nonneg=True)
        # Slack on v_max to keep the QP feasible when the brake cannot meet the
        # constraint instantly (transients on entering a speed-limit zone).
        self._v_slack = cp.Variable(N + 1, name="v_slack", nonneg=True)

        # Parameters (updated every tick)
        s0 = cp.Parameter(name="s0")
        v0 = cp.Parameter(name="v0")
        u_prev = cp.Parameter(name="u_prev")
        v_ref = cp.Parameter(N, name="v_ref")
        v_max = cp.Parameter(N + 1, nonneg=True, name="v_max")
        s_target = cp.Parameter(name="s_target")
        # Linearised residual params (scalars applied uniformly across horizon).
        a_v = cp.Parameter(name="a_v")          # coefficient on v_k
        a_thr = cp.Parameter(name="a_thr")
        a_brk = cp.Parameter(name="a_brk")
        offset = cp.Parameter(name="offset")

        u_applied = self._u_thr - self._u_brk

        # Cost (same as Traditional MPC)
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

        cons = [self._s[0] == s0, self._v[0] == v0]
        for k in range(N):
            cons.append(self._s[k + 1] == self._s[k] + dt * self._v[k])
            # Augmented dynamics with parameterised linearised NN residual:
            #   v_{k+1} = (1 + a_v) * v_k
            #            + (dt*b_dr + a_thr) * u_thr_k
            #            + (-dt*b_br + a_brk) * u_brk_k
            #            + (offset - dt*c)
            cons.append(
                self._v[k + 1]
                == (1.0 + a_v) * self._v[k]
                + (dt * b_dr + a_thr) * self._u_thr[k]
                + (-dt * b_br + a_brk) * self._u_brk[k]
                + (offset - dt * c)
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
            "a_v": a_v, "a_thr": a_thr, "a_brk": a_brk, "offset": offset,
        }

    def reset(self) -> None:
        self._u_thr_prev = 0.0
        self._u_brk_prev = 0.0

    # ------------------------------------------------------------------

    def _residual_linearisation(self, v: float, u_thr_prev: float, u_brk_prev: float,
                                  cr: float, payload: float, grade: float):
        """Query the NN for residual + Jacobian at the current state."""
        x = np.array([v, u_thr_prev, u_brk_prev, cr, payload, grade], dtype=np.float64)
        y, J = self._mlp.jacobian(x)
        y_val = float(y[0]) * self.cfg.residual_scale
        Jrow = J[0] * self.cfg.residual_scale
        a_v = float(Jrow[0])
        a_thr = float(Jrow[1])
        a_brk = float(Jrow[2])
        # offset such that linearisation passes through (current point, current y_val)
        offset = y_val - a_v * v - a_thr * u_thr_prev - a_brk * u_brk_prev
        return a_v, a_thr, a_brk, offset, y_val

    def step(
        self,
        v_meas: float,
        v_ref_now: float,
        dt: float,
        preview: Optional[Any] = None,
        s_meas: float = 0.0,
    ) -> ControlOutput:
        if preview is None or "v_ref" not in preview:
            raise ValueError("IMPCController.step requires preview['v_ref']")
        if abs(dt - self.mpc_dt) > 1e-6:
            raise ValueError(f"IMPC compiled for dt={self.mpc_dt}, called with dt={dt}")

        N = self.mpc.horizon
        v_ref_seq = np.asarray(preview["v_ref"], dtype=float)
        v_max_seq = np.asarray(preview.get("v_max", np.full(N + 1, 1e3)), dtype=float)
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

        cr = float(preview.get("surface_cr", 0.005))
        payload = float(preview.get("payload_kg", 0.0))
        grade = float(preview.get("grade_rad", 0.0))

        # NN linearisation
        a_v, a_thr, a_brk, offset, residual_val = self._residual_linearisation(
            v_meas, self._u_thr_prev, self._u_brk_prev, cr, payload, grade
        )

        # Push parameters
        self._params["s0"].value = s_meas
        self._params["v0"].value = v_meas
        self._params["u_prev"].value = self._u_thr_prev - self._u_brk_prev
        self._params["v_ref"].value = v_ref_seq
        self._params["v_max"].value = v_max_seq
        self._params["s_target"].value = s_target_val
        self._params["a_v"].value = a_v
        self._params["a_thr"].value = a_thr
        self._params["a_brk"].value = a_brk
        self._params["offset"].value = offset

        try:
            self._problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            self._u_thr_prev = 0.0
            self._u_brk_prev = 0.0
            return ControlOutput(u=0.0, diagnostics={"error": str(e), "status": "exception"})

        status = self._problem.status
        if (self._u_thr.value is None or self._u_brk.value is None
                or status not in ("optimal", "optimal_inaccurate")):
            u = 0.0
            u_thr0 = 0.0
            u_brk0 = 0.0
        else:
            u_thr0 = float(np.clip(self._u_thr.value[0], 0.0, 1.0))
            u_brk0 = float(np.clip(self._u_brk.value[0], 0.0, 1.0))
            u = float(np.clip(u_thr0 - u_brk0, -1.0, 1.0))

        self._u_thr_prev = u_thr0
        self._u_brk_prev = u_brk0
        return ControlOutput(
            u=u,
            diagnostics={
                "status": status,
                "residual_val": residual_val,
                "a_v": a_v, "a_thr": a_thr, "a_brk": a_brk, "offset": offset,
                "v_ref0": v_ref_seq[0],
                "v_max0": v_max_seq[0],
                "s_target": s_target_val,
            },
        )
