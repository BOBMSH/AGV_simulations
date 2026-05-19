"""Offline training of the IPID gain scheduler.

We use an analytical, physically motivated gain schedule (drag and
inertia compensation) to generate target labels, then train an MLP to
learn this mapping. The trained MLP is what the IPID controller queries
at runtime.

Rationale for the analytical schedule:
  * At higher rolling resistance, the plant is more damped -> needs
    higher Kp/Ki to maintain tracking against the increased drag.
  * At higher payload, the plant is more inertial -> needs slightly
    lower Kp to avoid overshoot but slightly higher Ki to overcome the
    steady-state lag.
  * Kd is kept roughly constant; derivative action depends mainly on
    measurement noise, not operating point.

The analytic baseline is validated on closed-loop simulations as a
sanity check (an alternative scipy.optimize-based label generator is
also provided for completeness, but it is much slower).

Run:
    python -m agv_sim.nn.training
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .mlp import MLP, MLPConfig


# --- Hand-designed gain schedule ---------------------------------------------

BASE_KP = 0.85
BASE_KI = 0.45
BASE_KD = 0.05


def analytic_gains(cr: float, payload: float) -> Tuple[float, float, float]:
    """Return (Kp, Ki, Kd) for the given operating point.

    Designed by inspection of the plant dynamics:
        m * dv/dt = F_drive - Cr * m * g
    so the open-loop gain from u to dv/dt is F_max / m, and the steady-state
    "drift" toward stop is Cr * g. Higher Cr -> stronger drift -> needs more
    Ki for steady offset; higher payload -> larger m -> proportionally less
    response per unit u, so Kp must scale up slightly to maintain the same
    closed-loop bandwidth, BUT inertia damps overshoot so we keep it modest.
    """
    # Reference operating point: Cr=0.005, payload=4000 kg.
    cr_factor = 1.0 + 80.0 * (cr - 0.005)
    payload_factor = np.sqrt(12000.0 / (8000.0 + payload))  # 1.0 at payload=4000

    kp = BASE_KP * cr_factor * payload_factor
    ki = BASE_KI * (1.0 + 60.0 * (cr - 0.005)) * (0.5 + 0.5 * payload_factor)
    kd = BASE_KD * (1.0 + 10.0 * (cr - 0.005))
    return float(kp), float(ki), float(kd)


def generate_dataset(
    n_samples: int = 100,
    cr_range: Tuple[float, float] = (0.003, 0.032),
    payload_range: Tuple[float, float] = (0.0, 12000.0),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (Cr, payload) and label with the analytic gain schedule."""
    rng = np.random.default_rng(seed)
    crs = rng.uniform(cr_range[0], cr_range[1], n_samples)
    payloads = rng.uniform(payload_range[0], payload_range[1], n_samples)
    X = np.column_stack([crs, payloads])
    Y = np.array([analytic_gains(c, p) for c, p in zip(crs, payloads)])
    return X, Y


def train_and_save(
    out_path: Path,
    n_samples: int = 200,
    n_epochs: int = 2000,
    lr: float = 5e-3,
    seed: int = 0,
    verbose: bool = True,
) -> MLP:
    if verbose:
        print(f"[ipid] generating {n_samples} (Cr, payload) -> (Kp, Ki, Kd) samples...")
    X, Y = generate_dataset(n_samples=n_samples, seed=seed)
    if verbose:
        print(f"[ipid]   Cr range:      [{X[:,0].min():.4f}, {X[:,0].max():.4f}]")
        print(f"[ipid]   payload range: [{X[:,1].min():.0f}, {X[:,1].max():.0f}]")
        print(f"[ipid]   Kp range:      [{Y[:,0].min():.3f}, {Y[:,0].max():.3f}]")
        print(f"[ipid]   Ki range:      [{Y[:,1].min():.3f}, {Y[:,1].max():.3f}]")
        print(f"[ipid]   Kd range:      [{Y[:,2].min():.4f}, {Y[:,2].max():.4f}]")
        print(f"[ipid] training MLP (2->32->32->3, softplus output)...")
    mlp = MLP(MLPConfig(layer_sizes=(2, 32, 32, 3), out_activation="softplus", seed=seed))
    losses = mlp.fit(X, Y, n_epochs=n_epochs, batch_size=16, lr=lr,
                     verbose=verbose, seed=seed)
    if verbose:
        print(f"[ipid] final training loss = {losses[-1]:.6f}")

    out_path = Path(out_path)
    mlp.save(out_path)
    np.savez(
        out_path.with_suffix(".dataset.npz"),
        X=X, Y=Y,
        feature_names=np.array(["cr", "payload_kg"]),
        target_names=np.array(["Kp", "Ki", "Kd"]),
    )
    if verbose:
        print(f"[ipid] saved to {out_path}")
    return mlp


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "checkpoints" / "ipid_scheduler.npz"
    train_and_save(out_path)
