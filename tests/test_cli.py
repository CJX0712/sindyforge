import os
import subprocess
import sys


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # 子进程不继承 conftest 的 sys.path 注入，显式补上仓库根以支持未安装场景
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [repo_root]
    )
    proc = subprocess.run(
        [sys.executable, "-m", "sindyforge.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc


def test_cli_quick_runs_and_writes(tmp_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = tmp_path / "bench.json"
    proc = _run_cli(["run", "--quick", "--out", str(out)], repo_root)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in data and "rows" in data
    assert len(data["rows"]) > 0
    # StabSINDy 必在方法聚合里
    assert "StabSINDy" in data["summary"]["methods"]


def test_top_level_api_is_importable_without_pythonpath():
    """打包契约：顶层 API 无需 PYTHONPATH 即可导入（README 示例依赖此约定）。"""
    import subprocess

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sindyforge as sf;"
            "assert sf.StabSINDy and sf.SindyPipeline and sf.SYSTEMS;"
            "assert sf.__version__ and sf.__author__;"
            "print('API-OK')",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "API-OK" in proc.stdout
