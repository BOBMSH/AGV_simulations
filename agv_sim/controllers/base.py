"""Controller abstract base class.

All four controllers (PID, IPID, MPC, IMPC) implement this interface so they
can be dropped into the same simulation loop interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ControlOutput:
    """Result of a controller step.

    Attributes
    ----------
    u : float
        Control input to the plant, in [-1, 1]. Positive = throttle fraction,
        negative = brake fraction.
    diagnostics : dict
        Controller-specific telemetry (e.g. current Kp/Ki/Kd for IPID, predicted
        horizon trajectory for MPC). Used by the live visualizer and KPI logger.
    """

    u: float
    diagnostics: dict


class Controller(ABC):
    """Abstract base for all controllers.

    Subclasses must implement step() and may override reset().
    """

    name: str = "base"

    @abstractmethod
    def step(
        self,
        v_meas: float,
        v_ref: float,
        dt: float,
        preview: Optional[Any] = None,
        s_meas: float = 0.0,
    ) -> ControlOutput:
        """Compute one control action.

        Parameters
        ----------
        v_meas : float
            Measured velocity at the current control tick [m/s].
        v_ref : float
            Reference velocity at the current control tick [m/s].
        dt : float
            Controller timestep [s].
        preview : optional dict
            Forward-looking information from the scenario. Has at minimum a
            "v_ref" array; may also have "v_max", "s_pred", "s_target".
            PID/IPID look only at v_ref[0]; MPC/IMPC consume the full dict.
        s_meas : float
            Measured position [m]. Used by MPC for the v_max(s) constraint
            and the position-stop terminal cost. PID/IPID ignore.
        """

    def reset(self) -> None:
        """Reset internal controller state. Default no-op."""
