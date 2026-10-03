# Changelog

All notable changes to this project are documented in this file.
格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循语义化版本。

## [0.1.0] - 2026-10-03

### 新增
- **StabSINDy 旗舰算法**：多 λ 稳定性选择 + 岭稳定 OLS 重拟合 + BIC（有效样本量校正）向后消元。
- **自适应 SG 导数窗口**（`data/derivatives.py`）：二阶差分稳健估计噪声 σ̂，
  用 SG 导数滤波器精确系数范数 ‖c_w‖ 计算噪声放大，取满足噪声预算的最小窗口。
  噪声 5% 下导数相对误差由 60%~167% 降至 9.4%。
- **平滑状态建库**：`estimate_derivative(..., return_smoothed=True)` 返回 SG 平滑状态，
  用于构建候选库，规避 x1³ 等非线性项的噪声放大（EIV）偏差。
- 候选库：3 阶多项式 + sin/cos（`domain/library.py`）。
- 基线方法：Lasso（LassoCV）、OLS、SingleSTR（消融对照）。
- 评估指标：rollout RMSE、support-F1、系数相对误差、**配对 t 检验**。
- CLI（`python -m sindyforge.cli run`）、端到端演示（`examples/run_demo.py`）。
- 离线降级：sklearn/scipy 缺失时自动切到纯 numpy 坐标下降 Lasso 与中心差分。
- 文档：`docs/architecture.md`、`docs/model_card.md`、README、CI、Dockerfile、Makefile。
- 14 项单测（含离线兜底路径、确定性、导数精度）。

### 变更
- Duffing 改为**保守双阱**（delta=0）并取大振幅轨道：带阻尼版本会衰减到不动点，
  使 x1 与截距项近共线，辨识问题退化不可解。
- 门禁改为三项（support-F1 / 稀疏度 / 系数精度），rollout RMSE 降级为诊断量——
  极限环系统上它由相位误差主导，不是合格判别量。
- 统计检验由 Welch 双样本改为**配对 t 检验**：同一 cell 内方法天然配对，
  系统间难度差异在作差时抵消（同一效应 p 从 0.317 变为 6.21e-39）。

### 修复
- 坐标下降 Lasso 发散：`b_old` 引用别名导致一次迭代即"收敛"；G-Jacobi 形式用更新后的
  `b[j]` 造成正反馈几何发散。改为残差增量式更新。
- `str_clean` 相对阈值被共线爆炸系数（±317）撑大，误删全部真值项。
  加岭稳定项 `ridge_rel=1e-6·mean(diag(ΘᵀΘ))` 压住系数幅值。
- λ 阈值以 `max|OLS|` 为基准时，被共线库撑大的伪影（1e5 量级）绑架。
  改用 LassoCV 自适应缩放参考 α。
- 噪声导数的 DC 偏置被常数项吸收并撑大阈值，导致真值项 freq=0。
  增加去均值 + 用平滑状态建库。
- `numpy_lasso` 惩罚空间与阈值空间不一致（惩罚在 L2 标准化空间、阈值用原始 OLS），
  阈值被放大 col_norm 倍误删真值项。
- `Config.to_dict()` 引用已删除的 `sign_frac` 字段导致序列化崩溃。
- CLI 缺少 `__main__.py` 无法 `python -m sindyforge.cli`。

### 已知边界
- 幅值仅为主导项 1/28 的弱真值项（Lorenz `ẋ2` 的 `x2=−1`）无法恢复，
  无噪下整体 F1 上限 0.9333。详见 `docs/model_card.md`。
