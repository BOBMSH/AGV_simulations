"""Train the IMPC plant-residual model.

Goal: learn a neural network f_theta(v, u_thr, u_brk, cr, payload, grade)
that captures the discrepancy between the IMPC's nominal linear plant
model and the true non-linear plant.

The nominal model used by Traditional MPC is:
    v_{k+1} = v_k + dt * (b_dr * u_thr - b_br * u_brk - c_nom)

with b_dr, b_br, c_nom fixed. It ignores:
  (a) motor first-order lag (force builds up over ~tau_m)
  (b) surface variation (true c_drag depends on Cr)
  (c) payload variation (true open-loop gain b = F_max/m depends on m)
  (d) road grade (gravity along-slope)

We collect (input, residual) pairs by simulating the true plant under
randomised operating conditions, and train an MLP to map input ->
residual. The IMPC linearises this residual at the current state each
tick and folds the result into its QP, keeping the MPC linear.

Run:
    python -m agv_sim.nn.impc_training
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..controllers.mpc import MPCPlantNominal
from ..plant import AGVParameters, AGVPlant, EnvironmentProfile
from .mlp import MLP, MLPConfig


def generate_residual_dataset(
    n_samples: int = 3000,
    cr_range: Tuple[float, float] = (0.003, 0.030),
    payload_range: Tuple[float, float] = (0.0, 18000.0),
    grade_range: Tuple[float, float] = (-0.07, 0.07),    # +/- 7% grade
    v_range: Tuple[float, float] = (0.0, 3.0),           # AGV speed range
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the true plant under random configurations and record residuals.

    Returns
    -------
    X : (n_samples, 6) array  - inputs [v, u_thr, u_brk, cr, payload, grade]
    Y : (n_samples, 1) array  - residual delta_v per controller tick
    """
    rng = np.random.default_rng(seed)
    params = AGVParameters()
    nom = MPCPlantNominal()
    dt_ctrl = params.dt_ctrl
    n_sub = max(1, int(round(dt_ctrl / params.dt_sim)))

    # Sample inputs.
    v0 = rng.uniform(v_range[0], v_range[1], n_samples)
    crs = rng.uniform(cr_range[0], cr_range[1], n_samples)
    payloads = rng.uniform(payload_range[0], payload_range[1], n_samples)
    grades = rng.uniform(grade_range[0], grade_range[1], n_samples)

    # Sample controls: in the IMPC the controls are split into thr/brk; we
    # mirror that by sampling either positive or negative u_total then
    # splitting.
    u_total = rng.uniform(-1.0, 1.0, n_samples)
    u_thr = np.maximum(u_total, 0.0)
    u_brk = np.maximum(-u_total, 0.0)

    Y = np.zeros((n_samples, 1))
    for i in range(n_samples):
        env = EnvironmentProfile(
            surface_cr=lambda s, t, c=crs[i]: c,
            grade_rad=lambda s, t, g=grades[i]: g,
            payload_kg=lambda s, t, p=payloads[i]: p,
            include_aero=True,    # include aero in plant (so MLP captures it as residual)
        )
        plant = AGVPlant(params, env, rng=np.random.default_rng(seed + i))
        plant.reset(v0=v0[i])
        # Start with motor force at steady-state for applied u (so residual
        # mostly captures the static mismatch, not motor lag transient).
        u_signed = u_thr[i] - u_brk[i]
        if u_signed >= 0:
            plant.state.f_drive = u_signed * params.f_drive_max
        else:
            plant.state.f_drive = u_signed * params.f_brake_max

        # Apply held control across dt_ctrl.
        for _ in range(n_sub):
            plant.step(u_signed, params.dt_sim)
        v1_true = plant.state.v
        delta_v_true = v1_true - v0[i]

        # Nominal model prediction over the same dt_ctrl.
        delta_v_nom = dt_ctrl * (nom.b_drive * u_thr[i] - nom.b_brake * u_brk[i] - nom.c_drag)
        v1_nom = v0[i] + delta_v_nom
        # Plant clips v at 0 -- mirror that in the nominal so residuals are
        # meaningful even near the floor.
        v1_nom = max(v1_nom, 0.0)
        delta_v_nom = v1_nom - v0[i]

        residual = delta_v_true - delta_v_nom
        Y[i, 0] = residual

    X = np.column_stack([v0, u_thr, u_brk, crs, payloads, grades])
    return X, Y


def train_and_save(
    out_path: Path,
    n_samples: int = 3000,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    seed: int = 0,
    verbose: bool = True,
) -> MLP:
    if verbose:
        print(f"[impc] generating {n_samples} (state, control, params) -> residual samples...")
    X, Y = generate_residual_dataset(n_samples=n_samples, seed=seed)
    if verbose:
        print(f"[impc]   v range:       [{X[:,0].min():.2f}, {X[:,0].max():.2f}]")
        print(f"[impc]   cr range:      [{X[:,3].min():.4f}, {X[:,3].max():.4f}]")
        print(f"[impc]   payload range: [{X[:,4].min():.0f}, {X[:,4].max():.0f}]")
        print(f"[impc]   grade range:   [{X[:,5].min():+.3f}, {X[:,5].max():+.3f}] rad")
        print(f"[impc]   residual range:[{Y.min():+.4f}, {Y.max():+.4f}]  std={Y.std():.4f}")
        print(f"[impc] training MLP (6->32->32->1, linear output)...")

    mlp = MLP(MLPConfig(layer_sizes=(6, 32, 32, 1), out_activation="linear", seed=seed))
    losses = mlp.fit(X, Y, n_epochs=n_epochs, batch_size=64, lr=lr,
                     normalise_x=True, normalise_y=True, verbose=verbose, seed=seed)
    if verbose:
        # Compare to a "predict zero" baseline.
        zero_mse = float(np.mean(Y ** 2))
        nn_mse = float(np.mean((mlp.forward(X) - Y) ** 2))
        print(f"[impc] final training loss (normalised) = {losses[-1]:.6f}")
        print(f"[impc] MSE on raw residuals: zero-baseline={zero_mse:.6f}, NN={nn_mse:.6f}  "
              f"(reduction {100*(1 - nn_mse/zero_mse):.1f}%)")

    out_path = Path(out_path)
    mlp.save(out_path)
    np.savez(
        out_path.with_suffix(".dataset.npz"),
        X=X, Y=Y,
        feature_names=np.array(["v", "u_thr", "u_brk", "cr", "payload_kg", "grade_rad"]),
        target_names=np.array(["delta_v_residual"]),
    )
    if verbose:
        print(f"[impc] saved to {out_path}")
    return mlp


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "checkpoints" / "impc_residual.npz"
    train_and_save(out_path)
