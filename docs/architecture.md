# 架构说明

## 分层与依赖方向

```
cli ──> pipeline ──> {data, discovery, eval} ──> core
                 └──> domain（候选库）
```

**单一无环**：`core` 不依赖任何上层；`domain` 只依赖 numpy；`eval` 可被任意层调用但不反向依赖。
每个 (系统, 噪声, seed, 方法) 组合产出一行 `BenchmarkRow`，`benchmark()` 汇总全部真实验行结果。

## 模块职责

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `sindyforge/core/config.py` | 全部可调参数 + ENV 覆盖 | 所有数值确定性、可复现 |
| `sindyforge/core/seed.py` | `set_all` 全局播种 | 保证逐位可复现 |
| `sindyforge/core/backends.py` | `available_sklearn()` / `available_scipy()` 探测 | 离线降级开关的唯一入口 |
| `sindyforge/core/types.py` | `DiscoveryResult` / `BenchmarkRow` / `SystemSpec` | 数据契约 |
| `sindyforge/data/systems.py` | 合成 DGP（Lorenz / VdP / Duffing / Pendulum）+ 真值支撑声明 | `true_terms` 特征名必须与 library 对齐 |
| `sindyforge/data/derivatives.py` | 自适应 SG 导数 + 噪声估计 | 见下文「四个关键设计」 |
| `sindyforge/domain/library.py` | 多项式/三角候选库 + 特征命名 | 命名与 `true_terms` 严格一致，否则 F1 恒为 0 |
| `sindyforge/discovery/strlasso.py` | **StabSINDy 旗舰** + numpy Lasso + OLS | 见下文 |
| `sindyforge/discovery/*_baseline.py` | Lasso / OLS / SingleSTR 基线与消融对照 | 与旗舰同接口 |
| `sindyforge/eval/metrics.py` | rollout RMSE / support-F1 / 系数误差 / 配对 t 检验 | 发散返回 1e6 哨兵值保持可比 |
| `sindyforge/pipeline/pipeline.py` | 编排 + 门禁聚合 | 门禁判定集中在此 |

## StabSINDy 流程

```
1. 去均值 Ẋ            —— 噪声导数的 DC 偏置会被常数项吸收，干扰选择
2. z-score 候选库 Θ      —— 惩罚在「解释方差」意义上均匀（实测关闭会崩，见下）
3. LassoCV 求参考 α_ref  —— 数据自适应缩放，绕开共线导致的 OLS 系数膨胀
4. 多 λ 路径（0.2×~5×）  —— 统计各特征被选中频率
5. 频率 ≥ 50% 取并集     —— 稳定性共识
6. 岭稳定 OLS 重拟合     —— ridge_rel=1e-6·mean(diag(ΘᵀΘ))，压住共线爆炸系数
7. BIC 向后消元          —— 有效样本量 n_eff 由残差 lag-1 自相关推得
8. 温和 STR 清洗         —— 阈值 3%·max|ξ|（仅清理数值噪声级残项）
```

## 四个关键设计（都是踩坑后加的）

### 1. 自适应 SG 窗口 —— 固定窗口必崩一端

实测（噪声 5%，dt=0.02）：

| 窗口 | VdP | Duffing | Pendulum | Lorenz |
|---|---|---|---|---|
| 7 | 101.6% | 59.9% | 166.7% | 17.3% |
| 21 | 18.1% | 10.7% | 29.8% | 23.8% |
| 41 | 10.2% | 6.4% | 16.6% | 45.8% |

VdP/Pendulum 噪声受限（要长窗口），Lorenz 偏置受限（要短窗口）——**没有通用固定窗口**。

做法（纯数据驱动，不用真值）：
1. 二阶差分 `Δ²x` 的 MAD 稳健估计噪声 σ̂（光滑信号被压到 O(h²x'')，只剩 6σ² 分量）；
2. 小窗口先求一次导数，其功率 = 信号功率 + σ̂²‖c‖²，反解出信号功率；
3. 取满足 `σ̂²‖c_w‖² ≤ τ²·信号功率`（τ=0.10）的**最小**窗口 —— 在满足噪声预算下最小化偏置。

结果：噪声 5% 下全系统一致降到 **9.4%**；噪声 10% 下 ~10%（原 120%~333%）。

### 2. 用平滑状态构建候选库，而非含噪状态

含噪状态让 `x1³` 把观测噪声放大 `3x1²·σ`。Duffing 上实测放大到 2.4，与真值信号同量级，
产生变量误差（EIV）偏差——**系数符号都会被带翻**（曾出现 `x1=-17.7, sin(x1)=+14.8` 的
共线抵消对）。改用 SG 平滑后的状态建库后，VdP 系数误差从 15%~25% 降到 **<3%**。

### 3. λ 网格以 LassoCV 自适应缩放

多项式+三角库高度共线（Duffing 上 corr(x1, x1³)=0.94），原始 OLS 系数被撑到 1e5 量级。
任何以 `max|OLS|` 为基准的固定阈值都会被这个伪影绑架：阈值被放大后把
`str_rel·max|ξ|` 撑到 9.5，真值项 x1、x1³ 全被误删。改用 LassoCV 求参考 α 后消失。

### 4. BIC 用有效样本量 n_eff，而非 N

SG 平滑让残差在约 w 个点内强相关，独立信息量远小于 N。直接用 N 会让 BIC 把
「只降低约 1% RSS」的冗余项判为显著——实测 VdP 的 ẋ1 方程多出一个 `x1²=0.03` 的伪项，
长期 rollout 因此系统性漂移。

`n_eff = N(1−ρ)/(1+ρ)`，ρ 取**残差**的 lag-1 自相关（不是信号 y 的——
光滑周期信号的 lag-1 自相关天然趋近 1，会撞到下限、丧失判别力）。
实测 n_eff ≈ 2N/w（VdP 47 / Duffing 61 / Pendulum 35）。

## 门禁设计：为什么不拿 rollout RMSE 当门禁

1. **混沌系统**：Lorenz 长程 rollout 必然指数发散，任何模型都失效。
2. **极限环系统**：VdP/Pendulum 的短程 rollout 由**周期/相位误差**主导。
   相位在极限环上中性稳定、误差线性累积，2% 的系数误差就能造成数倍 RMSE 差异，
   且随 seed 高度波动（实测 9 个非混沌 cell 中旗舰 5 胜 4 负）。
3. **密集模型占便宜**：Lasso（30.8 项）/ OLS（40.5 项）把平滑偏置吸收进额外项，
   在同初始条件的短程拟合上占优——这恰是 SINDy 要避免的过拟合。

故门禁改为系统辨识的真正目标三项：**support-F1 / 稀疏度 / 系数精度**，
rollout RMSE 仅作诊断量如实报告。

**统计上用配对 t 检验而非 Welch 双样本**：每个 (系统, 噪声, seed) cell 同时跑所有方法，
天然配对，系统间难度差异在作差时被抵消。用 Welch 会丢掉这一结构——
同一效应在 Welch 下 p=0.317（不显著），配对下 p=6.21e-39。

## 离线降级

`sindyforge/core/backends.py` 探测 sklearn / scipy 是否可用：

| 能力 | 首选 | 降级 |
|---|---|---|
| Lasso 求解 | `sklearn.linear_model.Lasso` | 自研坐标下降（残差增量式） |
| λ 参考值 | `LassoCV` | z 空间 OLS 最大系数 |
| 导数估计 | `scipy.signal.savgol_filter` | `np.gradient` 中心差分 |

`tests/test_offline_fallback.py` 用 monkeypatch 强制走降级路径验证。

## 确定性

`sindyforge/core/seed.set_all` 全局播种；全流程无未受控随机源（`make_dataset` 用 `RandomState(seed)`）。
`examples/run_demo.py` 的 `_determinism` 连跑两轮全量基准，要求方法聚合指标**逐位相等**
（当前 `exact_match=True`）。
