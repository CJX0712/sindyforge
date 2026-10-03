import os
import subprocess
import sys


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-m", "sindyforge.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc


def test_cli_quick_runs_and_writes(tmp_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
