"""SindyForge CLI：argparse 入口。

用法：
  python -m sindyforge.cli run [--quick] [--out benchmark.json]
  python -m sindyforge.cli run --systems lorenz pendulum --noise 0.05 0.10 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from .. import __version__
from ..core.config import Config
from ..core.seed import set_all
from ..data.systems import SYSTEMS
from ..pipeline.pipeline import SindyPipeline, summarize


def _reconfigure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _build_parser():
    p = argparse.ArgumentParser(
        prog="sindyforge",
        description="SindyForge · 稀疏非线性动力学辨识 (SINDy / StabSINDy)",
    )
    p.add_argument("--version", action="version", version=f"SindyForge {__version__}")
    sub = p.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="跑全量/快速基准")
    run.add_argument("--out", default="benchmark.json", help="结果 JSON 落盘路径")
    run.add_argument("--systems", nargs="*", default=None)
    run.add_argument("--noise", nargs="*", type=float, default=None)
    run.add_argument("--seeds", nargs="*", type=int, default=None)
    run.add_argument("--degree", type=int, default=None)
    run.add_argument("--no-trig", action="store_true", help="关闭三角函数候选")
    run.add_argument("--quick", action="store_true", help="小配置快速冒烟")
    return p


def main(argv=None):
    _reconfigure_stdout()
    args = _build_parser().parse_args(argv)
    if args.cmd is None:
        args = _build_parser().parse_args(["run"])
    set_all(getattr(args, "seed", 42) if False else 42)

    cfg = Config()
    if args.quick:
        cfg = Config(
            n_steps=600,
            horizon=150,
            noise_levels=(0.05,),
            seeds=(0, 1, 2),
            degree=args.degree or 3,
            include_trig=not args.no_trig,
        )
    else:
        if args.degree is not None:
            cfg.degree = args.degree
        cfg.include_trig = not args.no_trig

    systems = args.systems if args.systems else list(SYSTEMS.keys())
    for s in systems:
        if s not in SYSTEMS:
            print(f"[warn] 未知系统 {s}，已跳过", file=sys.stderr)
    systems = [s for s in systems if s in SYSTEMS]
    noises = tuple(args.noise) if args.noise else cfg.noise_levels
    seeds = tuple(args.seeds) if args.seeds else cfg.seeds

    pipe = SindyPipeline(config=cfg, methods=["StabSINDy", "Lasso", "OLS", "SingleSTR"])
    rows = pipe.benchmark(system_names=systems, noise_levels=noises, seeds=seeds)
    summ = summarize(rows)

    # 打印表
    print("=" * 78)
    print(f"SindyForge {__version__} · 方法聚合（mean±std，多 seed 真实运行）")
    print("=" * 78)
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
        print("-" * 78)
        print(f"门禁：旗舰 {g['flagship']} vs 强基线 {g['baseline']}（配对 t 检验）")
        f = g["support_f1"]
        print(
            f"  [1] support-F1      {f['flagship']:.4f} vs {f['baseline']:.4f}"
            f"  → {f['rel_gain'] * 100:+.1f}%  (p={f['p']:.3g})  "
            f"{'PASS' if f['passed'] else 'FAIL'}"
        )
        n = g["n_terms"]
        print(
            f"  [2] 稀疏度 n_terms  {n['flagship']:.1f} vs {n['baseline']:.1f}"
            f"  → {n['rel_gain'] * 100:+.1f}%  (p={n['p']:.3g})  "
            f"{'PASS' if n['passed'] else 'FAIL'}"
        )
        c = g["coef_err"]
        print(
            f"  [3] 系数相对误差    {c['flagship']:.4f} vs {c['baseline']:.4f}"
            f"  → {c['rel_gain'] * 100:+.1f}%  (p={c['p']:.3g})  "
            f"{'PASS' if c['passed'] else 'FAIL'}"
        )
        r = g["rollout_diagnostic"]
        print(
            f"  -- rollout RMSE（诊断，不入门禁）  {r['flagship_rmse']:.4f} vs"
            f" {r['baseline_rmse']:.4f}  cell 胜率 {r['cell_wins']}  (p={r['p']:.3g})"
        )
        print(f"  门禁总判定：{'PASS' if g['passed'] else 'FAIL'}")
    print("=" * 78)

    out = {
        "meta": {
            "tool": "SindyForge",
            "version": __version__,
            "author": "晨星",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": cfg.to_dict(),
            "systems": systems,
        },
        "summary": summ,
        "rows": [vars(r) for r in rows],
    }
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已落盘：{os.path.abspath(out_path)}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
