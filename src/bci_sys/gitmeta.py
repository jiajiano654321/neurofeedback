"""记录代码版本（D2）：git 短 hash + dirty 标记。非 git 环境返回 'unknown'。"""
from __future__ import annotations

import subprocess


def code_version() -> str:
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
