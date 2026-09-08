"""replay-monitor 命令行入口。"""
from __future__ import annotations

import argparse
import sys

from . import monitor


def main(argv=None) -> int:
    # 确保 stdout/stderr 使用 UTF-8，避免 Windows 控制台 GBK 编码导致 µ 等字符报错
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(prog="replay-monitor", description="1x 回放 ds007169")
    ap.add_argument("--subj", required=True, help="被试编号，如 001")
    ap.add_argument("--duration-sec", type=float, default=0.0,
                    help="回放时长（秒）；0=全量（默认）")
    ap.add_argument("--data-root", default=None, help="ds007169 根目录（默认项目内）")
    ap.add_argument("--out-dir", default="outputs", help="产物根目录")
    args = ap.parse_args(argv)

    try:
        status = monitor.run_replay(
            args.subj, duration_sec=args.duration_sec,
            root=args.data_root, out_root=args.out_dir,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        return 2

    if status == "aborted":
        print("\n[中止] 已写 run.json(status=aborted)，本次数据标记为中止。", file=sys.stderr)
        return 130
    print(f"[完成] 产物在 {args.out_dir}/replay/sub-{args.subj}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
