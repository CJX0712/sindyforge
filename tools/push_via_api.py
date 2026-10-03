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


def local_commits():
    """本地提交序列（旧 → 新），只取 git 跟踪文件构成的历史。"""
    shas = _git("rev-list", "--reverse", "HEAD").split()
    out = []
    for sha in shas:
        paths = _git("ls-tree", "-r", "--name-only", sha).splitlines()
        msg = _git("log", "-1", "--pretty=%B", sha).strip()
        author = _git("log", "-1", "--pretty=%an", sha).strip()
        email = _git("log", "-1", "--pretty=%ae", sha).strip()
        tree = _git("rev-parse", f"{sha}^{{tree}}").strip()
        out.append(
            {
                "sha": sha,
                "tree": tree,
                "paths": [p for p in paths if p],
                "msg": msg,
                "author": author,
                "email": email,
            }
        )
    return out


def remote_commits():
    """远端 main 的提交（新 → 旧转旧 → 新），并附上去重后的 tree SHA 集合。"""
    res = gh_api("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=100")
    out = [
        {
            "sha": c["sha"],
            "msg": c["commit"]["message"].strip(),
            "tree": c["commit"]["tree"]["sha"],
        }
        for c in res
    ]
    out.reverse()
    return out


def missing_commits(local, remote):
    """返回尚未出现在远端的本地提交（按 tree SHA 内容寻址判断，天然幂等）。

    git tree SHA 由内容唯一决定，因此「tree 已在远端出现」等价于「该提交内容已推送」，
    不受重复提交 / 顺序错乱影响。
    """
    seen = {c["tree"] for c in remote}
    return [c for c in local if c["tree"] not in seen]


def blob_for(rel, rev):
    """取某提交下某文件的内容字节；不存在则返回 None。"""
    res = subprocess.run(["git", "show", f"{rev}:{rel}"], capture_output=True, cwd=ROOT)
    if res.returncode != 0:
        return None
    return res.stdout


def build_tree(commit):
    items = []
    for rel in commit["paths"]:
        content = blob_for(rel, commit["sha"])
        if content is None:
            continue
        items.append(
            {
                "path": rel,
                "mode": "100644",
                "type": "blob",
                "content": content.decode("utf-8"),
            }
        )
    return gh_api("POST", f"/repos/{OWNER}/{REPO}/git/trees", {"tree": items})["sha"]


def push_history(rebuild=False):
    local = local_commits()
    remote = remote_commits()
    print(f"本地提交 {len(local)} 个；远端 {len(remote)} 个")

    todo = local if rebuild else missing_commits(local, remote)
    if not todo:
        print("远端已包含全部本地提交，无需推送")
        return

    if rebuild:
        print("重建模式：丢弃远端现有历史，重新镜像本地全部提交")
        parent = None  # 首个提交不设父提交，彻底切断旧历史
    else:
        parent = remote[-1]["sha"] if remote else bootstrap_parent()

    for i, c in enumerate(todo, 1):
        tree = build_tree(c)
        payload = {
            "message": c["msg"],
            "tree": tree,
            "author": {"name": c["author"] or "晨星", "email": c["email"]},
        }
        if parent:
            payload["parents"] = [parent]
        commit = gh_api("POST", f"/repos/{OWNER}/{REPO}/git/commits", payload)
        parent = commit["sha"]
        print(f"  [{i}/{len(todo)}] {c['sha'][:8]} {c['msg'].splitlines()[0][:60]}")

    gh_api(
        "PATCH",
        f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
        {"sha": parent, "force": True},
    )
    print(f"已更新 refs/heads/{BRANCH} → {parent[:8]}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="通过 Git Data API 推送本地历史")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="丢弃远端现有历史，从 bootstrap 提交重建完整镜像（用于清理被污染的历史）",
    )
    args = ap.parse_args()

    n = len([p for p in _git("ls-files", "-z").split("\0") if p])
    print(f"当前索引文件 {n} 个")
    push_history(rebuild=args.rebuild)
    print("完成：https://github.com/CJX0712/sindyforge")


if __name__ == "__main__":
    main()
