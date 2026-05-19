"""NumPy-only multilayer perceptron.

A small, self-contained MLP suitable for the IPID gain scheduler (Phase 3)
and the IMPC plant-residual model (Phase 4). NumPy keeps the project
dependency-light and avoids torch's install footprint.

Features:
  - Configurable layer sizes
  - ReLU hidden activations, configurable output activation
  - Adam optimizer
  - Save / load to .npz
  - Manual analytical gradient for output Jacobian wrt input (useful for IMPC
    SQP linearisation in Phase 4)

The optimisation target is supervised regression with mean-squared-error
loss, which is what we need for both phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _softplus(x: np.ndarray) -> np.ndarray:
    # numerically stable softplus
    return np.where(x > 20.0, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


def _softplus_grad(x: np.ndarray) -> np.ndarray:
    # d/dx softplus(x) = sigmoid(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


@dataclass
class MLPConfig:
    """Layer sizes (including input and output)."""
    layer_sizes: Tuple[int, ...] = (3, 32, 32, 3)
    out_activation: str = "softplus"  # "softplus", "linear"
    seed: int = 0


class MLP:
    """Fully-connected MLP with manual backprop and Adam optimiser."""

    def __init__(self, cfg: MLPConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(cfg.layer_sizes) - 1):
            fan_in = cfg.layer_sizes[i]
            fan_out = cfg.layer_sizes[i + 1]
            # He init for ReLU
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        # Adam moments (initialized lazily)
        self._m_W: List[np.ndarray] = [np.zeros_like(W) for W in self.W]
        self._v_W: List[np.ndarray] = [np.zeros_like(W) for W in self.W]
        self._m_b: List[np.ndarray] = [np.zeros_like(b) for b in self.b]
        self._v_b: List[np.ndarray] = [np.zeros_like(b) for b in self.b]
        self._step = 0
        # Input normalisation stats (set during fit)
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: Optional[np.ndarray] = None
        self.y_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_cached(self, x: np.ndarray):
        """Forward pass that also returns intermediate activations for backprop."""
        a = x  # (batch, in)
        zs = []   # pre-activations
        acts = [a]  # post-activations (incl input)
        n_layers = len(self.W)
        for li in range(n_layers):
            z = acts[-1] @ self.W[li] + self.b[li]
            zs.append(z)
            if li < n_layers - 1:
                a = _relu(z)
            else:
                if self.cfg.out_activation == "softplus":
                    a = _softplus(z)
                else:  # linear
                    a = z
            acts.append(a)
        return acts, zs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Predict y given input x (already normalised if normalisation was fit)."""
        if x.ndim == 1:
            x = x[None, :]
            single = True
        else:
            single = False
        x_in = x
        if self.x_mean is not None:
            x_in = (x_in - self.x_mean) / self.x_std
        acts, _ = self._forward_cached(x_in)
        y = acts[-1]
        if self.y_mean is not None:
            y = y * self.y_std + self.y_mean
        return y[0] if single else y

    # ------------------------------------------------------------------
    # Backprop + Adam
    # ------------------------------------------------------------------

    def _train_step(self, x_batch: np.ndarray, y_batch: np.ndarray, lr: float,
                    beta1=0.9, beta2=0.999, eps=1e-8) -> float:
        """One SGD step on a minibatch. Returns MSE loss."""
        acts, zs = self._forward_cached(x_batch)
        y_pred = acts[-1]
        diff = y_pred - y_batch  # (B, out)
        loss = float(np.mean(diff ** 2))
        # Output gradient
        n_layers = len(self.W)
        if self.cfg.out_activation == "softplus":
            d_out = (2.0 / diff.size) * diff * _softplus_grad(zs[-1])
        else:
            d_out = (2.0 / diff.size) * diff
        # Backprop
        grads_W: List[Optional[np.ndarray]] = [None] * n_layers
        grads_b: List[Optional[np.ndarray]] = [None] * n_layers
        delta = d_out
        for li in reversed(range(n_layers)):
            a_prev = acts[li]
            grads_W[li] = a_prev.T @ delta
            grads_b[li] = delta.sum(axis=0)
            if li > 0:
                delta = (delta @ self.W[li].T) * _relu_grad(zs[li - 1])
        # Adam update
        self._step += 1
        bc1 = 1.0 - beta1 ** self._step
        bc2 = 1.0 - beta2 ** self._step
        for li in range(n_layers):
            self._m_W[li] = beta1 * self._m_W[li] + (1 - beta1) * grads_W[li]
            self._v_W[li] = beta2 * self._v_W[li] + (1 - beta2) * (grads_W[li] ** 2)
            m_hat = self._m_W[li] / bc1
            v_hat = self._v_W[li] / bc2
            self.W[li] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            self._m_b[li] = beta1 * self._m_b[li] + (1 - beta1) * grads_b[li]
            self._v_b[li] = beta2 * self._v_b[li] + (1 - beta2) * (grads_b[li] ** 2)
            m_hat_b = self._m_b[li] / bc1
            v_hat_b = self._v_b[li] / bc2
            self.b[li] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
        return loss

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-3,
        normalise_x: bool = True,
        normalise_y: bool = False,
        verbose: bool = False,
        seed: int = 0,
    ) -> List[float]:
        """Supervised regression training. Returns the per-epoch loss curve.

        normalise_x: subtract mean and divide by std of inputs (recommended).
        normalise_y: subtract mean and divide by std of targets. Set to False
                     when using softplus output activation with strictly
                     positive targets, since z-scored targets would be half
                     negative and softplus cannot represent those.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if normalise_x:
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0) + 1e-8
            Xn = (X - self.x_mean) / self.x_std
        else:
            Xn = X
        if normalise_y:
            self.y_mean = Y.mean(axis=0)
            self.y_std = Y.std(axis=0) + 1e-8
            Yn = (Y - self.y_mean) / self.y_std
        else:
            Yn = Y
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        losses = []
        for ep in range(n_epochs):
            idx = rng.permutation(n)
            ep_losses = []
            for s in range(0, n, batch_size):
                batch = idx[s : s + batch_size]
                loss = self._train_step(Xn[batch], Yn[batch], lr=lr)
                ep_losses.append(loss)
            losses.append(float(np.mean(ep_losses)))
            if verbose and (ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1):
                print(f"  epoch {ep:4d}  loss={losses[-1]:.6f}")
        return losses



    def jacobian(self, x: np.ndarray) -> tuple:
        """Return (output, dy/dx) at a single input x.

        Computes the Jacobian via forward-mode differentiation through the
        normalisation + ReLU layers + output activation. Hand-derived so we
        do not need autograd. Returns:
            y_pred  : (out,) array
            jac     : (out, in) array, d y_unnorm / d x_unnorm

        The Jacobian accounts for x and y normalisation if those were fit.
        """
        if x.ndim == 1:
            x_in = x.copy()
        else:
            raise ValueError("jacobian expects a single 1-D input")
        # Normalise input
        if self.x_mean is not None:
            x_n = (x_in - self.x_mean) / self.x_std
        else:
            x_n = x_in
        # Forward + accumulate Jacobian wrt normalised input.
        a = x_n[None, :]                    # (1, in)
        # J_a: d a / d x_n. Start as identity (1, in, in)? Easier to track per-batch=1.
        # We'll keep J as (current_dim, in_dim).
        d = a.shape[1]
        J = np.eye(d)                       # (in, in) at the input layer
        n_layers = len(self.W)
        for li in range(n_layers):
            z = a @ self.W[li] + self.b[li]   # (1, out_li)
            if li < n_layers - 1:
                # ReLU activation
                mask = (z > 0).astype(z.dtype)
                # da/dx_n = mask * (W^T @ J(prev))
                # New J shape: (out_li, in)
                J = mask[0][:, None] * (self.W[li].T @ J)
                a = z * mask
            else:
                # Output activation
                if self.cfg.out_activation == 'softplus':
                    sig = 1.0 / (1.0 + np.exp(-np.clip(z[0], -50.0, 50.0)))
                    J = sig[:, None] * (self.W[li].T @ J)
                    a = _softplus(z)
                else:  # linear
                    J = self.W[li].T @ J
                    a = z
        y_norm = a[0]                        # (out,)
        # Account for input scaling: d y_norm / d x_unnorm = (d y_norm / d x_norm) * (1/x_std)
        if self.x_mean is not None:
            J = J / self.x_std[None, :]
        # Account for output unnormalisation
        if self.y_mean is not None:
            y = y_norm * self.y_std + self.y_mean
            J = J * self.y_std[:, None]
        else:
            y = y_norm
        return y, J

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {f"W{i}": W for i, W in enumerate(self.W)}
        data.update({f"b{i}": b for i, b in enumerate(self.b)})
        if self.x_mean is not None:
            data["x_mean"] = self.x_mean
            data["x_std"] = self.x_std
        if self.y_mean is not None:
            data["y_mean"] = self.y_mean
            data["y_std"] = self.y_std
        data["layer_sizes"] = np.array(self.cfg.layer_sizes, dtype=np.int64)
        data["out_activation"] = np.array([self.cfg.out_activation], dtype="U16")
        np.savez(path, **data)

    @classmethod
    def load(cls, path: Path) -> "MLP":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        layer_sizes = tuple(int(x) for x in data["layer_sizes"])
        out_activation = str(data["out_activation"][0])
        cfg = MLPConfig(layer_sizes=layer_sizes, out_activation=out_activation)
        mlp = cls(cfg)
        n_layers = len(layer_sizes) - 1
        mlp.W = [data[f"W{i}"] for i in range(n_layers)]
        mlp.b = [data[f"b{i}"] for i in range(n_layers)]
        if "x_mean" in data.files:
            mlp.x_mean = data["x_mean"]
            mlp.x_std = data["x_std"]
        if "y_mean" in data.files:
            mlp.y_mean = data["y_mean"]
            mlp.y_std = data["y_std"]
        return mlp
