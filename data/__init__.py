"""data 层：合成动力学系统 + 导数估计。"""

from .derivatives import estimate_derivative
from .systems import (
    DUFFING,
    LORENZ,
    PENDULUM,
    SYSTEMS,
    VANDERPOL,
    System,
    make_dataset,
    rollout_truth,
    simulate,
)

__all__ = [
    "System",
    "SYSTEMS",
    "simulate",
    "make_dataset",
    "rollout_truth",
    "LORENZ",
    "VANDERPOL",
    "DUFFING",
    "PENDULUM",
    "estimate_derivative",
]
