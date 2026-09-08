"""theta-probe 命令行入口。"""
from __future__ import annotations

import argparse
import sys

from . import probe


def main(argv=None) -> int:
    # 确保 stdout/stderr 使用 UTF-8，避免 Windows 控制台 GBK 编码问题
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(prog="theta-probe", description="离线 Fz theta 负荷检验")
    ap.add_argument("--subj", help="单被试编号，如 001（硬门控，失败退出码 1）")
    ap.add_argument("--all", action="store_true", help="跑推荐被试汇总（参考，退出码恒 0）")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args(argv)

    if not args.subj and not args.all:
        ap.error("必须指定 --subj 或 --all")

    try:
        if args.all:
            rows = probe.run_all(root=args.data_root, out_dir=args.out_dir)
            for r in rows:
                med = f"{r['m1']:.1f} < {r['m2']:.1f} < {r['m3']:.1f} < {r['m4']:.1f}"
                if r["pass"]:
                    print(f"sub-{r['subj']}: PASS ({med})")
                else:
                    print(f"sub-{r['subj']}: FAIL ({med})")
            print(f"[完成] 汇总在 {args.out_dir}/theta/all/（参考，不作硬门控）")
            return 0

        ok = probe.run_probe(args.subj, root=args.data_root, out_dir=args.out_dir)
        if ok:
            print(f"PASS: sub-{args.subj} theta 1→4-back 单调上升")
            print(f"[完成] 产物在 {args.out_dir}/theta/{args.subj}/")
            return 0
        print(f"FAIL: sub-{args.subj} theta 不单调——检查 theta 计算或通道映射，不许进 IT3")
        return 1
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
