# SindyForge

> 稀疏非线性动力学系统辨识（SINDy 范式）+ 稳定性选择增强的 **StabSINDy** 旗舰算法。
> 从含噪观测轨迹中恢复**可解释的、稀疏的**微分方程右端。

[![CI](https://github.com/CJX0712/sindyforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/sindyforge/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Author: 晨星](https://img.shields.io/badge/author-晨星-orange.svg)](https://github.com/CJX0712)

---

## 一句话

给一条含噪轨迹，SindyForge 还你一个**写得出来的方程**——而不是一个黑箱。

## 为什么是它

系统辨识的产出不是「预测得多准」，而是「方程对不对」。SindyForge 把
**稳定性选择（stability selection）** 接进 SINDy 范式，在 5%~10% 观测噪声下
显著优于 Lasso / OLS / 单阈值 STR 基线：

| 方法 | support-F1 ↑ | 系数相对误差 ↓ | 平均项数 ↓ | rollout RMSE（诊断） |
|---|---|---|---|---|
| **StabSINDy（旗舰）** | **0.7842** | **0.3375** | **5.5** | 0.0374 |
| Lasso（强基线） | 0.2232 | 1.1637 | 30.8 | 55556.27 |
| OLS | 0.1831 | 3.8199 | 40.5 | 0.0561 |
| SingleSTR（消融对照） | 0.6722 | 0.3835 | 7.3 | 0.1094 |

> 数据来自 `examples/benchmark.json`（4 系统 × 2 噪声 × 3 seed × 4 方法 = 96 行真实运行，
> 无任何手填）。门禁用**配对 t 检验**（同一 cell 内方法对齐后作差）。

三项门禁全部通过（全部为 StabSINDy vs Lasso 配对检验）：

| 门禁 | StabSINDy | Lasso | 提升 | p 值 | 判定 |
|---|---|---|---|---|---|
| support-F1 | 0.7842 | 0.2232 | **+251.3%** | 6.21e-39 | ✅ PASS |
| 稀疏度（项数） | 5.46 | 30.75 | **+82.2%** | 6.63e-10 | ✅ PASS |
| 系数相对误差 | 0.3375 | 1.1637 | **+71.0%** | 0.0113 | ✅ PASS |

消融（剥离稳定性共识，退化为单阈值 STR）：support-F1 **+16.7%**，p=0.00537 ——
证明提升来自稳定性共识这一设计，而非调参。

## 快速开始

```bash
pip install -e ".[dev]"

# 跑默认基准（约 1 分钟），产出 benchmark.json
python -m sindyforge.cli run

# 极快冒烟
python -m sindyforge.cli run --quick

# 端到端演示：确定性校验 + 消融 + 失败案例 → benchmark.json / analysis.json
python examples/run_demo.py --mode default
```

Python API：

```python
import numpy as np
from sindyforge.core.config import Config
from sindyforge.data.systems import LORENZ, make_dataset
from sindyforge.data.derivatives import estimate_derivative
from sindyforge.domain.library import PolynomialLibrary
from sindyforge.discovery.strlasso import StabSINDy

cfg = Config()
clean, noisy, _ = make_dataset(LORENZ, cfg.n_steps, cfg.dt, 0.05, 0)
Xdot, Xs = estimate_derivative(noisy, cfg.dt, return_smoothed=True)
lib = PolynomialLibrary(LORENZ.dim, degree=3, include_trig=True)
res = StabSINDy(cfg).discover(lib.build(Xs), Xdot)
for j, name in enumerate(lib.feature_names):
    if abs(res.xi[j, 0]) > 1e-12:
        print(f"dx1/dt += {res.xi[j, 0]:+.4f} * {name}")
```

## 方法：StabSINDy 做了什么

```
含噪轨迹 X ──① 自适应 SG 平滑 ──> 平滑状态 Xs + 导数 Ẋ
                                        │
     Xs ──② 多项式/三角候选库 Θ ────────┤
                                        ▼
     ③ 多 λ Lasso 路径 → 各特征「被选中频率」
     ④ 频率 ≥ 50% 取稳定性并集
     ⑤ 岭稳定 OLS 重拟合 + BIC（有效样本量校正）向后消元
                                        ▼
                                 稀疏系数矩阵 Ξ
```

四个关键设计，每一个都是踩坑后加上的（详见 `docs/architecture.md`）：

1. **自适应 SG 窗口** —— 固定窗口必崩一端：窗口 7 时 VdP/Pendulum 的导数相对误差
   达 100%~167%，窗口 41 时 Lorenz 的截断偏置又涨到 46%。
   做法：二阶差分稳健估计噪声 σ̂，用 SG 导数滤波器的精确系数范数 ‖c_w‖ 算噪声放大，
   取满足「导数噪声 ≤ 10% 导数信号」的**最小**窗口。
   实测噪声 5% 下误差从 60%~167% 降到 **9.4%**（全系统一致）。
2. **用平滑状态构建候选库** —— 含噪状态会让 x1³ 把噪声放大 3x1²·σ（Duffing 实测 2.4，
   与真值信号同量级），造成变量误差（EIV）偏差，系数符号都会被带翻。
3. **λ 网格以 LassoCV 自适应缩放** —— 多项式+三角库高度共线，原始 OLS 系数被撑到
   1e5 量级，任何以 `max|OLS|` 为基准的固定阈值都会被这个伪影绑架。
4. **BIC 用有效样本量而非 N** —— SG 平滑让残差在约 w 个点内强相关，直接用 N=600
   会把「只降低 1% RSS」的冗余项判为显著。

## 已知边界（如实记录）

- **幅值悬殊的弱真值项难以恢复**：Lorenz 的 `ẋ2 = 28·x1 − 1·x2 − 1·x1x3` 中，
  x2 的系数只有主导项的 1/28，在任何 λ 网格/阈值组合下都进不了稳定性并集。
  无噪场景下该维度 F1 为 0.8（整体 0.9333），这是当前方法的硬边界。
- **短程 rollout RMSE 不是合格判别量**：极限环系统（VdP/Pendulum）的短程 rollout
  由**周期/相位误差**主导，相位在极限环上中性稳定、误差线性累积，2% 的系数误差
  就能造成数倍 RMSE 差异，且随 seed 高度波动。故本仓库**不将其设为门禁**，
  仅作为诊断量如实报告（当前 17/18 cell 胜出，p=0.317）。
- **混沌系统不参与 rollout 门禁**：Lorenz 长程 rollout 必然指数发散，这是动力学
  固有属性而非方法缺陷。

完整说明见 `docs/model_card.md`。

## 项目结构

```
sindyforge/
├── core/        config · seed（确定性）· errors · types · interfaces · backends（离线探测）
├── data/        systems（合成 DGP + 真值支撑）· derivatives（自适应 SG）
├── domain/      library（多项式/三角候选库 + 特征命名对齐）
├── discovery/   strlasso（StabSINDy 旗舰）· lasso_baseline · ols_baseline · single_threshold
├── eval/        metrics（rollout RMSE / support-F1 / 系数误差 / 配对 t 检验）
├── pipeline/    pipeline（编排 + 门禁聚合）
├── cli/         命令行入口
├── examples/    run_demo.py（端到端验收）
└── tests/       14 项单测（含离线兜底路径）
```

## 离线可跑

sklearn / scipy 缺失时自动降级到纯 numpy 路径（`core/backends.py` 探测）：

- Lasso → 自研坐标下降（残差增量式，保证收敛）
- Savitzky-Golay → 中心差分

`tests/test_offline_fallback.py` 用 monkeypatch 强制走降级路径验证。

## 复现

```bash
make install   # pip install -e ".[dev]"
make test      # pytest
make lint      # ruff check + format --check
make bench     # 默认基准
make demo      # 端到端演示
make ci        # lint + test + bench 冒烟
```

## 许可

MIT © 晨星
