"""discovery 层：动力学辨识器集合。"""

from .lasso_baseline import LassoBaseline
from .ols_baseline import OLSBaseline
from .single_threshold import SingleThresholdSTR
from .strlasso import StabSINDy, numpy_lasso, ols_fit

ALL_METHODS = {
    "StabSINDy": StabSINDy,
    "Lasso": LassoBaseline,
    "OLS": OLSBaseline,
    "SingleSTR": SingleThresholdSTR,
}

__all__ = [
    "StabSINDy",
    "LassoBaseline",
    "OLSBaseline",
    "SingleThresholdSTR",
    "numpy_lasso",
    "ols_fit",
    "ALL_METHODS",
]
