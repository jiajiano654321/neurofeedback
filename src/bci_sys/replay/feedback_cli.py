"""feedback-loop 命令行入口（spec §5.2/§6/§8）。

live：preflight+读盘（rc=2 在起服务器前）→ 绑 127.0.0.1 的 uvicorn →
点确认后引擎线程起跑；引擎自然终态由 watcher 关服务器；Ctrl-C → abort 后 rc=130。
replay：不起引擎、不写任何文件。
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from .. import config, io_bids
from . import feedback_engine as eng
from . import webapp

_ALL_CH = (config.TARGET.channel, *config.EOG_PROXY_CHANNELS)


def _watch_runner_then_stop_server(server, runner, app, *,
                                   terminal_linger_sec: float = 10.0) -> None:
    """等 runner 真正 start（_t 被赋值）后再 join，终态后等 WS 断开再关服务器。

    为什么要等 start：api_start 在 runner_factory 返回之后才调 runner.start()，
    而 factory 里就起了 watch 守护线程。若 watch 在 start 前调 join，
    FeedbackRunner.join 因 self._t is None 会**立即**返回 → 服务器还没开跑就被
    should_exit 关掉 → CLI 兜底把 outcome 还 None 的 runner abort 掉，
    引擎刚启动在 k=0 就落 user_abort（启动竞态，uvicorn 真实路径每次必现）。

    寿命语义：引擎终态 且 无观看者 WS 连接，服务器才退出。
    - 正常路径：前端收到 final 后主动关 WS → watcher 检测到 ws_taken=False → 退出
    - 兜底：终态后最多再等 terminal_linger_sec（防死页面/永不关连接），
      到时仍有 WS 连着也强退；graceful_shutdown 再兜 uvicorn 层的底。
    """
    # 轮询等 _t 被赋值（start() 才会做）；间隔 10ms，最多等 30s（极端慢启动兜底）
    started = False
    for _ in range(3000):
        if runner._t is not None:
            started = True
            break
        time.sleep(0.01)
    if not started:
        print("[watcher] runner 在 30s 内未 start，按未启动关停服务器",
              file=sys.stderr)
    runner.join()  # 超时未 start 时 join 必立即返回，保留统一出口
    # 引擎已终态：等 WS 断开再退出（观看者在，服务器就得在）
    # 用真实 time.sleep（墙钟策略，不与注入的 _sleep_fn 快放时钟混用）
    linger_deadline = time.time() + terminal_linger_sec
    while getattr(app.state, "ws_taken", False):
        if time.time() >= linger_deadline:
            print(f"[watcher] 终态驻留 {terminal_linger_sec}s 上限到，"
                  f"强制关停（WS 仍连着）", file=sys.stderr)
            break
        time.sleep(0.05)
    # uvicorn.Server.should_exit 是公开文档化的退出信号（serve 循环每轮检查），
    # 置位即令主线程 server.run() 返回；本线程只发信号，不碰引擎落盘
    server.should_exit = True


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
        self._t = None  # 与 FeedbackRunner 接口对齐：start 前为 None

    def start(self, **_k):
        self._t = object()  # 标记已 start（watcher 轮询等的就是这个）
        self.outcome = {"status": "aborted", "reason": "engine_error",
                        "run_dir": self.run_dir}

    def join(self, timeout=None):
        # 与 FeedbackRunner.join 同构：start() 前 _t is None 直接返回；
        # start() 同步置终态，故 start 后 join 也立即返回
        if self._t is None:
            return


def run(argv=None, *, _server_cls=None, _sleep_fn=None, _clock=None,
        _fault=None, _terminal_linger_sec: float | None = None,
        _graceful_shutdown_sec: float | None = None) -> int:
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
        threading.Thread(
            target=_watch_runner_then_stop_server,
            args=(server, r, app_state["app"]),
            kwargs={"terminal_linger_sec": terminal_linger_sec},
            daemon=True,
        ).start()
        return r

    terminal_linger_sec = _terminal_linger_sec if _terminal_linger_sec is not None else 10.0
    graceful_shutdown_sec = _graceful_shutdown_sec if _graceful_shutdown_sec is not None else 5.0
    app_state: dict = {}
    app = webapp.create_app(subj=args.subj, data_root=args.data_root,
                            out_root=args.out_dir, runner_factory=runner_factory)
    app_state["app"] = app
    server = _make_server(app, args.port, _server_cls,
                          graceful=graceful_shutdown_sec)
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


def _make_server(app, port: int, server_cls, *, graceful: float = 5.0):
    if server_cls is not None:
        return server_cls(app, port)
    import uvicorn
    config_uv = uvicorn.Config(app, host="127.0.0.1", port=port,
                               log_level="warning",
                               timeout_graceful_shutdown=graceful)
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
