"""Scenarios package."""

from .base import Scenario, ScenarioConstraints
from .scenario_a_simple import ScenarioA
from .scenario_b_friction import ScenarioB
from .scenario_c_dock import ScenarioC
from .scenario_d_combined import ScenarioD

SCENARIOS = {
    "A": ScenarioA,
    "B": ScenarioB,
    "C": ScenarioC,
    "D": ScenarioD,
}

__all__ = ["Scenario", "ScenarioConstraints",
           "ScenarioA", "ScenarioB", "ScenarioC", "ScenarioD", "SCENARIOS"]
