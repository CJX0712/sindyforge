"""SindyForge 端到端演示与验收分析。

产出（均来自真实运行，禁止手填）：
  - benchmark.json：逐行结果 + 方法聚合 + 门禁
  - analysis.json ：确定性二次校验 / 消融 / 失败案例

用法：
  python examples/run_demo.py            # 默认中等配置（用于交付报告）
  python examples/run_demo.py --quick    # 极快冒烟（CI 用）
  python examples/run_demo.py --full     # 大配置
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sindyforge.core.config import Config
from sindyforge.core.seed import set_all
from sindyforge.data.systems import SYSTEMS
from sindyforge.eval.metrics import paired_ttest
from sindyforge.pipeline.pipeline import SindyPipeline, summarize

METHODS = ["StabSINDy", "Lasso", "OLS", "SingleSTR"]


def _cfg(mode: str) -> Config:
    if mode == "quick":
        return Config(
            n_steps=500,
            horizon=120,
            noise_levels=(0.05,),
            seeds=(0, 1, 2),
            degree=3,
            include_trig=True,
        )
    if mode == "full":
        return Config(
            n_steps=2000,
            horizon=400,
            noise_levels=(0.02, 0.05, 0.10, 0.15),
            seeds=(0, 1, 2, 3, 4),
            degree=3,
            include_trig=True,
        )
    # 默认中等
    return Config(
        n_steps=1200,
        horizon=200,
        noise_levels=(0.05, 0.10),
        seeds=(0, 1, 2),
        degree=3,
        include_trig=True,
    )


def _run(cfg: Config):
    pipe = SindyPipeline(cfg, methods=METHODS)
    rows = pipe.benchmark()
    return rows, summarize(rows)


def _determinism(cfg: Config):
    set_all(cfg.seed)
    _, s1 = _run(cfg)
    set_all(cfg.seed)
    _, s2 = _run(cfg)
    mismatch = []
    for m in METHODS:
        if m not in s1["methods"] or m not in s2["methods"]:
            continue
        for key in ("rollout_rmse_mean", "support_f1_mean", "n_terms_mean"):
            v1 = s1["methods"][m][key]
            v2 = s2["methods"][m][key]
            if v1 != v2:  # 逐位一致（浮点同运算应逐位相等）
                mismatch.append((m, key, v1, v2))
    return {
        "exact_match": len(mismatch) == 0,
        "mismatches": mismatch,
        "flagship_rmse_run1": s1["methods"]["StabSINDy"]["rollout_rmse_mean"],
        "flagship_rmse_run2": s2["methods"]["StabSINDy"]["rollout_rmse_mean"],
    }


def _ablation(rows):
    """消融：旗舰 StabSINDy（含稳定性共识）vs 单阈值 STR（剥离稳定性共识）。

    主指标用 support-F1（系统辨识的核心目标），而非 rollout RMSE——
    后者在极限环系统上由相位误差主导，不反映消融的真实效应。
    """
    key = lambda r: (r.system, r.noise, r.seed)  # noqa: E731
    fb = {key(r): r for r in rows if r.method == "StabSINDy"}
    ab = {key(r): r for r in rows if r.method == "SingleSTR"}
    shared = sorted(set(fb) & set(ab))
    if not shared:
        return {"error": "无配对样本"}
    a = [fb[k].support_f1 for k in shared]
    b = [ab[k].support_f1 for k in shared]
    t, p, ma, mb = paired_ttest(a, b)
    na = [float(fb[k].n_terms) for k in shared]
    nb = [float(ab[k].n_terms) for k in shared]
    return {
        "metric": "support_f1",
        "flagship_f1": float(ma),
        "ablated_f1": float(mb),
        "rel_gain": float((ma - mb) / (abs(mb) + 1e-12)),
        "flagship_n_terms": float(np.mean(na)),
        "ablated_n_terms": float(np.mean(nb)),
        "t": float(t),
        "p": float(p),
        "n_pairs": len(shared),
    }


def _failures(rows):
    """失败案例 ≥3 条，全部从真实 results 派生，含原因归因。"""
    cases = []

    # 1) 高噪声下旗舰支撑未满分（漏/多检）
    miss = [r for r in rows if r.method == "StabSINDy" and r.support_f1 < 0.999]
    if miss:
        m = min(miss, key=lambda r: r.support_f1)
        cases.append(
            {
                "type": "support_incomplete",
                "system": m.system,
                "noise": m.noise,
                "seed": m.seed,
                "support_f1": m.support_f1,
                "n_terms": m.n_terms,
                "reason": "噪声抬高导数估计误差，稳定性共识在临界弱项上保守剔除（误伤真项）或保留噪声伪项。",
            }
        )

    # 2) 混沌系统 rollout 对初值敏感（Lorenz 大 RMSE）
    lorenz = [r for r in rows if r.system == "lorenz" and r.method == "StabSINDy"]
    if lorenz:
        worst = max(lorenz, key=lambda r: r.rollout_rmse)
        cases.append(
            {
                "type": "chaotic_sensitivity",
                "system": worst.system,
                "noise": worst.noise,
                "seed": worst.seed,
                "rollout_rmse": worst.rollout_rmse,
                "reason": "Lorenz 为混沌系统，即便系数近似正确，有限步 rollout 仍指数发散；这是动力学固有属性，非方法缺陷。",
            }
        )

    # 3) OLS 稠密过拟合导致 rollout 爆炸（反例说明稀疏必要）
    ols = [r for r in rows if r.method == "OLS"]
    if ols:
        blow = max(ols, key=lambda r: r.rollout_rmse)
        cases.append(
            {
                "type": "dense_overfit_blowup",
                "system": blow.system,
                "noise": blow.noise,
                "seed": blow.seed,
                "n_terms": blow.n_terms,
                "rollout_rmse": blow.rollout_rmse,
                "reason": "OLS 全特征稠密拟合（含噪声伪项），rollout 数值发散；凸显稀疏回归的必要性。",
            }
        )

    return cases[:3] if cases else [{"type": "none", "reason": "未捕获到典型失败案例"}]


def main(argv=None):
    ap = argparse.ArgumentParser(description="SindyForge demo + 验收分析")
    ap.add_argument("--mode", choices=["quick", "default", "full"], default="default")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args(argv)

    cfg = _cfg(args.mode)
    set_all(cfg.seed)
    rows, summ = _run(cfg)

    det = _determinism(cfg)
    abl = _ablation(rows)
    fails = _failures(rows)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    bench_path = os.path.join(out_dir, "benchmark.json")
    analysis_path = os.path.join(out_dir, "analysis.json")

    meta = {
        "tool": "SindyForge",
        "version": "0.1.0",
        "author": "晨星",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "config": cfg.to_dict(),
        "systems": list(SYSTEMS.keys()),
    }
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "summary": summ, "rows": [vars(r) for r in rows]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "determinism": det, "ablation": abl, "failures": fails},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 控制台摘要
    print("=" * 80)
    print(f"SindyForge 0.1.0 · 模式={args.mode} · 作者=晨星")
    print("=" * 80)
    print(
        f"{'method':<12}{'rolloutRMSE':>13}{'supportF1':>11}{'coefErr':>10}{'n_terms':>9}{'sec':>8}"
    )
    for m, s in summ["methods"].items():
        print(
            f"{m:<12}{s['rollout_rmse_mean']:>13.4f}{s['support_f1_mean']:>11.4f}"
            f"{s['coef_err_mean']:>10.4f}{s['n_terms_mean']:>9.1f}{s['elapsed_mean']:>8.3f}"
        )
    if "gate" in summ:
        g = summ["gate"]
        print("-" * 80)
        print(f"门禁（配对 t 检验）  {g['flagship']} vs {g['baseline']}")
        for gk, label in (
            ("support_f1", "support-F1  "),
            ("n_terms", "稀疏度      "),
            ("coef_err", "系数相对误差"),
        ):
            d = g[gk]
            print(
                f"  {label} {d['flagship']:.4f} vs {d['baseline']:.4f}"
                f"  → {d['rel_gain'] * 100:+.1f}%  p={d['p']:.3g}"
                f"  {'PASS' if d['passed'] else 'FAIL'}"
            )
        r = g["rollout_diagnostic"]
        print(
            f"  [诊断·不入门禁] rollout RMSE {r['flagship_rmse']:.4f} vs"
            f" {r['baseline_rmse']:.4f}  cell 胜率 {r['cell_wins']}  p={r['p']:.3g}"
        )
        print(f"  门禁总判定：{'PASS' if g['passed'] else 'FAIL'}")
    print(f"确定性逐位一致: {det['exact_match']}")
    print(
        f"消融 (StabSINDy vs 单阈值 STR, 按 support-F1): "
        f"{abl['rel_gain'] * 100:+.1f}%  p={abl['p']:.3g}"
    )
    print(f"失败案例: {len(fails)} 条")
    print(f"落盘: {bench_path}")
    print(f"落盘: {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
