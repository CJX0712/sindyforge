"""配置：ENV_SINDFORGE_* 覆盖 + schema 校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """系统级配置。所有数值均为确定性、可复现的输入参数。"""

    seed: int = 42
    n_steps: int = 1500  # 辨识用轨迹长度
    dt: float = 0.02  # 仿真步长
    horizon: int = 300  # rollout 预测步数
    noise_levels: tuple = (0.05, 0.10)  # 相对每维标准差的噪声比例
    seeds: tuple = (0, 1, 2)  # 多 seed 复现
    degree: int = 3  # 多项式库阶数
    include_trig: bool = True  # 是否加入 sin/cos 候选
    str_rel: float = 0.03  # STR 阈值 = str_rel * max|OLS|（温和，保留真值小项）
    stab_grid: tuple = (0.2, 0.5, 1.0, 2.0, 5.0)  # 相对参考 α 的倍数网格
    stab_frac: float = 0.5  # 稳定性并集门槛：出现在 ≥50% 的 λ 上才进入候选
    lasso_alphas: tuple = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

    @classmethod
    def from_env(cls) -> "Config":
        kw = {}
        mapping = {
            "SINDFORGE_SEED": ("seed", int),
            "SINDFORGE_N_STEPS": ("n_steps", int),
            "SINDFORGE_DT": ("dt", float),
            "SINDFORGE_HORIZON": ("horizon", int),
            "SINDFORGE_DEGREE": ("degree", int),
        }
        for env_key, (attr, caster) in mapping.items():
            v = os.environ.get(env_key)
            if v is not None:
                try:
                    kw[attr] = caster(v)
                except ValueError:
                    pass
        return cls(**kw)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "horizon": self.horizon,
            "noise_levels": list(self.noise_levels),
            "seeds": list(self.seeds),
            "degree": self.degree,
            "include_trig": self.include_trig,
            "str_rel": self.str_rel,
            "stab_grid": list(self.stab_grid),
            "stab_frac": self.stab_frac,
        }
