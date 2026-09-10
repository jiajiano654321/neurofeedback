"""qc-monitor 命令行入口。退出码：ok=0；preflight/数据错误=2；中止=130。"""
from __future__ import annotations

import argparse
import sys

from . import qc_monitor


def main(argv=None) -> int:
    # Windows 控制台 GBK 下 µ/✓/✗ 需要 UTF-8（与 replay CLI 同）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        prog="qc-monitor",
        description="1x 回放 + 实时三级质量判定（L1 残差 / L3 EOG 否决）",
    )
    ap.add_argument("--subj", required=True, help="被试编号，如 001")
    ap.add_argument("--duration-sec", type=float, default=0.0,
                    help="回放时长（秒）；0=全量（默认）")
    ap.add_argument("--data-root", default=None, help="ds007169 根目录（默认项目内）")
    ap.add_argument("--out-dir", default="outputs", help="产物根目录")
    args = ap.parse_args(argv)

    try:
        status = qc_monitor.run_qc(
            args.subj, duration_sec=args.duration_sec,
            root=args.data_root, out_root=args.out_dir,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        return 2

    if status == "aborted":
        print("\n[中止] 已写 run.json(status=aborted)，本次数据标记为中止。", file=sys.stderr)
        return 130
    print(f"[完成] 产物在 {args.out_dir}/qc/sub-{args.subj}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())