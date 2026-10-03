"""全局确定性入口。所有随机源一次设齐，保证同 seed 两次运行逐位一致。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_all(seed: int) -> int:
    """固定 seed，一次性设齐 Python / numpy 全局 + 哈希种子。

    返回 seed 本身（链式调用友好）。
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def make_rng(seed: int) -> np.random.RandomState:
    """显式返回一个可复现的 RandomState（避免依赖全局状态）。"""
    return np.random.RandomState(int(seed))
