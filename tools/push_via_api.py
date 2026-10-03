"""通过 GitHub Git Data API 推送本地仓库（本地 git push 被代理屏蔽时的三级降级路径）。

为什么需要它：本环境的 HTTP 代理允许 api.github.com / codeload.github.com，
但屏蔽 github.com，导致 `git push`（smart HTTP 走 github.com）返回 502。
改用 Git Data API 建树 → 建提交 → 建引用，全程只访问 api.github.com。
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER, REPO, BRANCH = "CJX0712", "sindyforge", "main"
SKIP_DIRS = {".git", "__pycache__", ".ruff_cache", ".pytest_cache"}


def collect(root):
    """只取 git 跟踪的文件 —— 临时诊断脚本/缓存天然被排除在外。"""
    res = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, cwd=root)
    rels = [p for p in res.stdout.split("\0") if p]
    out = []
    for rel in rels:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            out.append({"path": rel, "mode": "100644", "type": "blob", "content": f.read()})
    return out


def gh_api(method, endpoint, payload=None, tolerant=False):
    # Windows 下 stdin 传 JSON 会被 gh 解析失败，改走 --input 临时文件（ensure_ascii 转义非 ASCII）
    cmd = ["gh", "api", "-X", method, endpoint]
    tmp = None
    if payload is not None:
        tmp = os.path.join(ROOT, "_api_payload.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
        cmd += ["--input", tmp]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if tmp:
        os.remove(tmp)
    if res.returncode != 0:
        if tolerant:
            return None
        print(f"[FAIL] {method} {endpoint}\n{res.stderr[:2000]}", file=sys.stderr)
        sys.exit(1)
    return json.loads(res.stdout) if res.stdout.strip() else {}


def bootstrap_parent():
    """空仓库上 trees API 会 409（尚无默认分支）。用 contents API 建初始提交确立 HEAD。"""
    import base64

    ref = gh_api("GET", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}", tolerant=True)
    if ref is not None:
        return ref["object"]["sha"]
    print("仓库为空 → 用 contents API 建立初始提交以确立默认分支")
    r = gh_api(
        "PUT",
        f"/repos/{OWNER}/{REPO}/contents/README.md",
        {
            "message": "chore: 初始化仓库",
            "content": base64.b64encode("# sindyforge\n".encode("utf-8")).decode(),
            "branch": BRANCH,
        },
    )
    return r["commit"]["sha"]


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", cwd=ROOT
    ).stdout


def git_message():
    """直接复用本地 HEAD 的提交信息，避免本地与远端描述漂移。"""
    return _git("log", "-1", "--pretty=%B").strip()


def git_author():
    name = _git("config", "user.name").strip() or "晨星"
    email = _git("config", "user.email").strip() or "CJX0712@users.noreply.github.com"
    return {"name": name, "email": email}


def main():
    files = collect(ROOT)
    print(f"收集文件 {len(files)} 个")
    payload = {"tree": files}
    with open(os.path.join(ROOT, "_tree.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"tree payload: {os.path.getsize(os.path.join(ROOT, '_tree.json')) / 1024:.0f} KB")

    # 空仓库上 trees API 会 409，必须先确立默认分支
    parent = bootstrap_parent()

    tree = gh_api("POST", f"/repos/{OWNER}/{REPO}/git/trees", payload)
    print("tree sha:", tree["sha"])

    msg = git_message()
    print("提交信息取自本地 git HEAD：", msg.splitlines()[0])

    # 仓库已由 bootstrap 建了初始提交，这里以它为父提交（保持线性历史）
    commit = gh_api(
        "POST",
        f"/repos/{OWNER}/{REPO}/git/commits",
        {
            "message": msg,
            "tree": tree["sha"],
            "parents": [parent],
            "author": git_author(),
        },
    )
    print("commit sha:", commit["sha"])

    gh_api(
        "PATCH",
        f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
        {"sha": commit["sha"], "force": True},
    )
    print(f"已更新 refs/heads/{BRANCH}")

    os.remove(os.path.join(ROOT, "_tree.json"))
    print("完成：https://github.com/CJX0712/sindyforge")


if __name__ == "__main__":
    main()
