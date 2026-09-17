"""feedback-loop 命令行入口（spec §5.2/§6/§8）。

live：preflight+读盘（rc=2 在起服务器前）→ 绑 127.0.0.1 的 uvicorn →
点确认后引擎线程起跑；引擎自然终态由 watcher 关服务器；Ctrl-C → abort 后 rc=130。
replay：不起引擎、不写任何文件。
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from .. import config, io_bids
from . import feedback_engine as eng
from . import webapp

_ALL_CH = (config.TARGET.channel, *config.EOG_PROXY_CHANNELS)


def _rc(status: str | None, reason: str | None) -> int:
    """spec §5.2 退出码映射。"""
    if status == "ok":
        return 0
    if status == "aborted":
        if reason == "user_abort":
            return 130
        if reason in ("incomplete_data", "wall_cap", "freeze_failed"):
            return 3
        return 1  # engine_error / 未知
    return 1  # 无终态兜底


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="feedback-loop",
        description="IT4 神经反馈会话（S3 冻结 → 6×120s block；web 可视化）",
    )
    ap.add_argument("--subj", default="001", help="被试编号（默认 001）")
    ap.add_argument("--data-root", default=None, help="ds007169 根目录（默认项目内）")
    ap.add_argument("--out-dir", default="outputs", help="产物根目录")
    ap.add_argument("--port", type=int, default=8000, help="本机端口（默认 8000）")
    ap.add_argument("--replay", default=None, metavar="PATH",
                    help="回放模式：run 目录或 feedback.jsonl 路径（只读，不起引擎）")
    return ap


class _FaultRunner:
    """测试专用替身（_fault 注入）：start 即 engine_error 终态，不读盘不写盘。"""

    def __init__(self, *_a, **_k):
        self.abort = threading.Event()
        self.outcome = None
        self.run_dir = ""

    def start(self, **_k):
        self.outcome = {"status": "aborted", "reason": "engine_error",
                        "run_dir": self.run_dir}

    def join(self, timeout=None):
        pass


def run(argv=None, *, _server_cls=None, _sleep_fn=None, _clock=None,
        _fault=None) -> int:
    args = _build_parser().parse_args(argv)

    # ---------------- replay：无引擎、无写入 ----------------
    if args.replay is not None:
        p = Path(args.replay)
        if p.is_dir():
            p = p / "feedback.jsonl"
        try:
            p = p.resolve(strict=True)
            webapp.load_replay(p)  # 先验证可读/首行 meta
        except (OSError, ValueError) as e:
            print(f"PREFLIGHT FAIL: replay 文件不可用：{e}", file=sys.stderr)
            return 2
        app = webapp.create_app(mode="replay", replay_file=p)
        server = _make_server(app, args.port, _server_cls)
        print(f"[feedback-loop] replay（只读） http://127.0.0.1:{args.port}  {p}")
        server.run()
        return 0  # 回放查看器退出一律 0（没有可中止的会话）

    # ---------------- live：preflight/读盘先于服务器（rc=2） ----------------
    chans = None
    if _fault != "engine_error":  # 测试替身路径不读盘（验证 rc=1 映射用）
        try:
            io_bids.preflight(args.subj, args.data_root)
            chans = io_bids.load_channels_uV(args.subj, list(_ALL_CH), args.data_root)
        except (FileNotFoundError, ValueError, KeyError, OSError) as e:
            print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
            return 2

    registry: dict = {}

    def runner_factory(*, subj, root, out_root):
        if _fault == "engine_error":
            r = _FaultRunner()
        else:
            r = eng.FeedbackRunner(subj=subj, root=root, out_root=out_root,
                                   chans=chans)
            # 测试注入快放时钟（生产路径两个都是 None → 真墙钟）
            r._sleep_fn, r._clock = _sleep_fn, _clock
        registry["r"] = r
        def watch():
            r.join()
            # uvicorn.Server.should_exit 是公开文档化的退出信号（serve 循环每轮检查），
            # 置位即令主线程 server.run() 返回；本线程只发信号，不碰引擎落盘
            server.should_exit = True
        threading.Thread(target=watch, daemon=True).start()
        return r

    app = webapp.create_app(subj=args.subj, data_root=args.data_root,
                            out_root=args.out_dir, runner_factory=runner_factory)
    server = _make_server(app, args.port, _server_cls)
    print(f"[feedback-loop] http://127.0.0.1:{args.port}  "
          f"（仅本机；浏览器关闭不中止会话）")
    server.run()

    runner = registry.get("r")
    if runner is not None and runner.outcome is None:
        # uvicorn 自己吃掉了 Ctrl-C（serve 正常返回但引擎还活着）：
        # 必须显式 abort 并等它落 aborted 行，不能让 daemon 线程被进程带走
        print("\n[中止] 等待引擎落 aborted 行…", file=sys.stderr)
        runner.abort.set()
        runner.join(timeout=10)
    outcome = getattr(runner, "outcome", None)
    if outcome is None:
        return _rc(None, None)
    if outcome["status"] == "ok":
        print(f"[完成] 产物在 {outcome['run_dir']}")
    else:
        print(f"\n[中止] reason={outcome.get('reason')}；数据已留痕不删除（D3）。",
              file=sys.stderr)
    return _rc(outcome["status"], outcome.get("reason"))


def _make_server(app, port: int, server_cls):
    if server_cls is not None:
        return server_cls(app, port)
    import uvicorn
    config_uv = uvicorn.Config(app, host="127.0.0.1", port=port,
                               log_level="warning")
    return uvicorn.Server(config_uv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows GBK 控制台 µ/中文
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
