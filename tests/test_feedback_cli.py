"""IT4 CLI：argparse/help、rc 映射、preflight 先于服务器、快放全跑 rc=0、
abort=130、截断=3、engine_error=1、replay 只读零写入。"""
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bci_sys import io_bids
from bci_sys.replay import feedback_cli as cli
from bci_sys.replay import feedback_engine as eng


class FakeClock:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, dt): self.t += dt


class _Server:
    """替身：不绑端口，用 TestClient 完成 POST /api/start 后等引擎收尾。"""
    def __init__(self, app, port, action=None):
        self.app, self.port, self.action = app, port, action

    def run(self):
        with TestClient(self.app) as c:
            r = c.post("/api/start")
            assert r.status_code == 200, r.text
            runner = self.app.state.runner
            if self.action == "abort":
                runner.abort.set()
            runner.join(30)


class _ReplayServer:
    def __init__(self, app, port, **k): self.app = app
    def run(self):
        with TestClient(self.app) as c:
            info = c.get("/api/run").json()
            r = c.get(f"/api/replay?file={info['replay_file']}")
            assert r.status_code == 200 and r.json()["frames"]


def _live_args(tmp, data_root):
    return ["--subj", "001", "--data-root", str(data_root), "--out-dir", str(tmp)]


def test_help_subprocess():
    r = subprocess.run([sys.executable, "-m", "bci_sys.replay.feedback_cli", "--help"],
                       capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert r.returncode == 0
    assert "feedback-loop" in r.stdout and "--replay" in r.stdout and "--port" in r.stdout


def test_bad_subj_rc2_before_server_and_zero_artifacts(tmp_path, data_root):
    before = sorted(p.name for p in tmp_path.glob("**/*"))
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.feedback_cli",
         "--subj", "999", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 2
    assert "PREFLIGHT FAIL" in r.stderr
    assert not (tmp_path / "feedback").exists()
    assert sorted(p.name for p in tmp_path.glob("**/*")) == before


def test_rc_map():
    assert cli._rc("ok", None) == 0
    assert cli._rc("aborted", "user_abort") == 130
    for reason in ("incomplete_data", "wall_cap", "freeze_failed"):
        assert cli._rc("aborted", reason) == 3
    assert cli._rc("aborted", "engine_error") == 1
    assert cli._rc(None, None) == 1  # 无终态（被强杀兜底）


def test_live_full_run_rc0(data_root, tmp_path, capsys):
    clk = FakeClock()
    rc = cli.run(_live_args(tmp_path, data_root),
                 _server_cls=_Server, _sleep_fn=clk.sleep, _clock=clk.now)
    assert rc == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:" in out  # spec §6：只绑本机且终端打印 URL
    runs = list((tmp_path / "feedback" / "sub-001").iterdir())
    rd = runs[0]
    run = json.loads((rd / "run.json").read_text("utf-8"))
    assert run["status"] == "ok" and run["n_running_seg"] == 3600


def test_user_abort_rc130(data_root, tmp_path):
    # abort 在 POST 返回后的下一条语句立即置位：引擎线程刚启动尚在首段，
    # 而 S3/S4 循环每段都检查 abort（快放全程需数秒 CPU，不可能在这一窗口内跑完）
    clk = FakeClock()
    rc = cli.run(_live_args(tmp_path, data_root),
                 _server_cls=lambda app, port: _Server(app, port, action="abort"),
                 _sleep_fn=clk.sleep, _clock=clk.now)
    assert rc == 130


def test_incomplete_data_rc3(data_root, tmp_path, monkeypatch):
    orig = io_bids.load_channels_uV
    def truncated(subj, names, root=None):
        # 2000 段(400s 数据)：够校准 600 段 + 跑完第 1 个运行块 600 段，
        # 之后才耗尽 → 终止原因必须是 S4 的 incomplete_data（rc=3），
        # 而非校准期 freeze_failed 或全程正常 ok
        return {c: a[:2000 * 50] for c, a in orig(subj, names, root).items()}
    monkeypatch.setattr(cli.io_bids, "load_channels_uV", truncated)
    clk = FakeClock()
    rc = cli.run(_live_args(tmp_path, data_root),
                 _server_cls=_Server, _sleep_fn=clk.sleep, _clock=clk.now)
    assert rc == 3
    rd = next((tmp_path / "feedback" / "sub-001").iterdir())
    run = json.loads((rd / "run.json").read_text("utf-8"))
    assert run["status"] == "aborted" and run["reason"] == "incomplete_data"


def test_engine_error_rc1(tmp_path):
    # 不走数据：fault 注入替身 runner，start 即终态 engine_error
    rc = cli.run(["--subj", "001", "--out-dir", str(tmp_path)],
                 _server_cls=_Server, _fault="engine_error")
    assert rc == 1


def test_replay_rc0_and_writes_nothing(data_root, tmp_path):
    # 先快放产出一次 run，再对它的 jsonl 起 replay 服务
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="rr", sleep_fn=clk.sleep, clock=clk.now)
    jf = Path(out["run_dir"]) / "feedback.jsonl"
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    rc = cli.run(["--replay", str(jf), "--port", "8123"], _server_cls=_ReplayServer)
    assert rc == 0
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert before == after  # replay 零写入


def test_bad_replay_path_rc2(tmp_path):
    rc = cli.run(["--replay", str(tmp_path / "nope.jsonl")], _server_cls=_ReplayServer)
    assert rc == 2


def test_watch_runner_waits_for_start_before_joining():
    """修复前竞态：watch 在 runner.start() 前就 join（_t is None → 立即返回），
    导致 server.should_exit 被提前置 True，服务器还没开始就退出。
    修复后 _watch_runner_then_stop_server 必须轮询等 _t 被赋值才 join。"""
    import types

    class StubRunner:
        def __init__(self):
            self._t = None
            self._started = threading.Event()
            self._done = threading.Event()

        def start(self):
            # 模拟 start() 给 _t 赋值（真实实现是起线程）
            self._t = object()
            self._started.set()

        def join(self, timeout=None):
            # 只有 _t 非 None 时才会阻塞等 done（和真实 join 行为一致）
            if self._t is None:
                return
            self._done.wait(timeout)

    runner = StubRunner()
    server = types.SimpleNamespace(should_exit=False)

    t = threading.Thread(target=cli._watch_runner_then_stop_server,
                         args=(server, runner), daemon=True)
    t.start()

    # start() 之前：watcher 必须卡在轮询，绝不置 should_exit
    # 旧实现（watch 直接 join）会在这里立即置 True → 红相
    import time
    time.sleep(0.05)
    assert server.should_exit is False, (
        "watcher 在 runner.start() 前就置了 should_exit —— 启动竞态未修复")

    # 现在真正 start，再释放 join
    runner.start()
    runner._done.set()

    # 2s 内轮询 should_exit 变 True
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if server.should_exit:
            break
        time.sleep(0.01)
    assert server.should_exit is True, "watcher 结束后未置 should_exit"
    t.join(timeout=1)
    assert not t.is_alive()
