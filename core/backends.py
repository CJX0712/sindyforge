"""后端可用性探测。SOTA 后端（scipy / sklearn）不可用时自动降级 Tier-1。"""

from __future__ import annotations


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def available_scipy() -> bool:
    try:
        import scipy  # noqa: F401

        return True
    except Exception:
        return False
