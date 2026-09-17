"""IT4 Web 层（spec §6）：静态页、单连接 WS+snapshot、start/abort、429/409、
replay 白名单与帧字段一致、reducer/seek、断 WS 引擎继续。"""
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from bci_sys.replay import feedback_engine as eng
from bci_sys.replay import webapp


class FakeRunner:
    """测试桩：与 FeedbackRunner 同形；脚本线程按开关发帧，abort 后收尾。"""

    def __init__(self, subj="001", root=None, out_root="outputs", run_id=None, chans=None):
        self.subj, self.root, self.out_root = subj, root, out_root
        self.run_id = run_id or "fake"
        self.run_dir = str(Path(out_root) / "feedback" / f"sub-{subj}" / self.run_id)
        self.bus = eng.EventBus()
        self.bus.snapshot = eng.initial_snapshot()
        self.abort = threading.Event()
        self.outcome = None
        self._go = threading.Event()
        self._done = threading.Event()

    def start(self, **kw):
        threading.Thread(target=self._script, daemon=True).start()

    def join(self, timeout=None):
        self._done.wait(timeout)

    @property
    def alive(self):
        return self.outcome is None

    def _pub(self, i):
        row = {"type": "seg", "k": 600 + i, "t1": round(120.2 + 0.2 * i, 6),
               "block_idx": 0, "seg_in_block": i, "theta_pow_uV2": 1.0 + i,
               "smoothed_uV2": 2.0, "hit": i % 2 == 0, "fz_clean_p2p_uv": 5.0,
               "eog_p2p_uv": 6.0, "valid": True, "reasons": []}
        snap = {**eng.initial_snapshot(), "phase": "block", "block_idx": 0,
                "seg_in_phase": i + 1, "k": 600 + i, "t1": row["t1"],
                "n_valid": i + 1, "n_hit": (i + 2) // 2}
        self.bus.publish(row, snap)

    def _script(self):
        self._go.wait(5.0)
        for i in range(30):
            if self.abort.is_set():
                break
            self._pub(i)
            time.sleep(0.005)
        if self.abort.is_set():
            self.outcome = {"status": "aborted", "reason": "user_abort",
                            "run_dir": self.run_dir}
        self._done.set()

    def release(self):
        self._go.set()


@pytest.fixture
def client():
    runner_holder = {}

    def factory(*a, **k):
        runner_holder["r"] = FakeRunner(*a, **k)
        return runner_holder["r"]

    app = webapp.create_app(runner_factory=factory)
    c = TestClient(app)
    yield c, runner_holder


def test_static_index_and_app_js_served_no_external_urls():
    app = webapp.create_app()
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "<html" in r.text.lower()
        rj = c.get("/app.js")
        assert rj.status_code == 200
        body = r.text + rj.text
        assert "http://" not in body and "https://" not in body  # 无外链/CDN


def test_run_state_before_start(client):
    c, _ = client
    r = c.get("/api/run")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "live" and body["status"] is None and body["run_dir"] is None
    # 确认页所需静态信息在引擎启动前即可回显（纯字符串拼接，不读盘）
    assert body["subj"] == "001"
    assert body["data_vhdr"] == "sub-001_task-nback_eeg.vhdr"
    assert body["target"] == {"channel": "Fz", "band_hz": [4.0, 8.0], "direction": "up"}


def test_start_creates_runner_and_second_start_is_409(client):
    c, holder = client
    r = c.post("/api/start")
    assert r.status_code == 200 and r.json()["run_dir"].endswith("sub-001/fake")
    holder["r"].release()
    assert c.post("/api/start").status_code == 409
    holder["r"].join(5)


def test_abort_202_while_running_409_before_start_and_after_terminal(client):
    c, holder = client
    assert c.post("/api/abort").status_code == 409  # 引擎不存在
    c.post("/api/start")
    holder["r"].release()
    assert c.post("/api/abort").status_code == 202
    holder["r"].join(5)
    assert c.get("/api/run").json()["status"] == "aborted"
    assert c.post("/api/abort").status_code == 409  # 已终态


def test_ws_snapshot_first_single_connection_429(client):
    c, holder = client
    with c.websocket_connect("/ws") as ws:
        env = ws.receive_json()
        assert env["msg"]["type"] == "snapshot"
        assert env["msg"]["mode"] == "live"
        assert env["msg"]["params"]["n_blocks"] == 6
        with pytest.raises(WebSocketDisconnect) as ei:
            with c.websocket_connect("/ws") as ws2:
                ws2.receive_json()
        assert ei.value.code == 429
        # 第一连接仍可用：start 后能逐帧收到
        c.post("/api/start")
        holder["r"].release()
        f = ws.receive_json()
        assert f["msg"]["type"] == "seg" and f["seq"] == 1
    holder["r"].join(5)


def test_ws_disconnect_does_not_stop_engine_and_reconnect_resyncs(client):
    c, holder = client
    c.post("/api/start")
    runner = holder["r"]
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        runner.release()
        ws.receive_json()  # 至少一帧
    # 断开后引擎照跑：等脚本发完 30 帧
    for _ in range(100):
        if runner.bus.seq >= 30:
            break
        time.sleep(0.02)
    assert runner.bus.seq == 30
    with c.websocket_connect("/ws") as ws2:
        env = ws2.receive_json()  # 重连 snapshot 直接追平到最新
        assert env["seq"] == 30
        assert env["msg"]["k"] == 629 and env["msg"]["n_valid"] == 30


def _real_run_jsonl(data_root, tmp_path, rid="fixed"):
    class Clk:
        t = 0.0
        def now(self): return self.t
        def sleep(self, dt): self.t += dt
    clk = Clk()
    eng.run_feedback(subj="001", root=data_root, out_root=tmp_path, run_id=rid,
                     sleep_fn=clk.sleep, clock=clk.now)
    rd = tmp_path / "feedback" / "sub-001" / rid
    return rd / "feedback.jsonl"


def test_replay_payload_matches_jsonl_and_offsets_monotonic(data_root, tmp_path):
    jf = _real_run_jsonl(data_root, tmp_path)
    app = webapp.create_app(mode="replay", replay_file=jf)
    with TestClient(app) as c:
        r = c.get(f"/api/replay?file={jf.name}")
        assert r.status_code == 200
        pl = r.json()
        rows = [json.loads(x) for x in jf.read_text("utf-8").splitlines()]
        assert pl["meta"] == rows[0]
        assert pl["frames"] == rows[1:]  # 数据帧与 jsonl 一字不差
        assert [f["type"] for f in pl["frames"][:600]] == ["calib_seg"] * 600
        offs = pl["offsets_ms"]
        assert all(offs[i + 1] >= offs[i] for i in range(len(offs) - 1))
        # 关键墙钟点：冻结 120s；block1 起点 数据240s+卡3s+rest30s=273s；rest 整 30s
        fi = next(i for i, f in enumerate(pl["frames"]) if f["type"] == "freeze")
        assert offs[fi] == 120_000
        bs1 = next(i for i, f in enumerate(pl["frames"])
                   if f["type"] == "phase" and f["phase"] == "block_start"
                   and f["block_idx"] == 1)
        # 数据 240s + 冻结卡 3s + 一个 rest 30s = 273s 墙钟
        assert offs[bs1] == 273_000
        rs = next(i for i, f in enumerate(pl["frames"])
                  if f["type"] == "phase" and f["phase"] == "rest_start")
        re_ = next(i for i, f in enumerate(pl["frames"])
                   if f["type"] == "phase" and f["phase"] == "rest_end")
        assert offs[re_] - offs[rs] == 30_000
        # 白名单：别的文件名 / 目录穿越一律 400
        assert c.get("/api/replay?file=other.jsonl").status_code == 400
        assert c.get("/api/replay?file=..%2f..%2fother.jsonl").status_code == 400
        # /api/run 在 replay 下回显白名单文件名与数据文件
        run_info = c.get("/api/run").json()
        assert run_info["mode"] == "replay"
        assert run_info["replay_file"] == jf.name
        assert run_info["data_vhdr"] == "sub-001_task-nback_eeg.vhdr"
        # replay 下写操作与 live WS 全拒
        assert c.post("/api/start").status_code == 403
        assert c.post("/api/abort").status_code == 403
        with pytest.raises(WebSocketDisconnect) as ei:
            with c.websocket_connect("/ws") as ws:
                ws.receive_json()
        assert ei.value.code == 1008
        return pl


def test_replay_reducer_finished_snapshot_and_seek(data_root, tmp_path):
    jf = _real_run_jsonl(data_root, tmp_path, rid="fixed2")
    rows = [json.loads(x) for x in jf.read_text("utf-8").splitlines()]
    pl = webapp.load_replay(jf)
    end = webapp.reduce_snapshot(pl["meta"], pl["frames"], len(pl["frames"]))
    assert end["phase"] == "finished" and end["status"] == "ok"
    assert len(end["block_rates"]) == 6
    assert end["n_valid"] == rows[-1]["n_valid"] and end["n_hit"] == rows[-1]["n_hit"]
    # seek：拖到墙钟 274s（block1 起点 273s 之后 1s）→ 落在 block_start 1 边界
    idx = webapp.boundary_before(pl["frames"], pl["offsets_ms"], 274_000)
    snap = webapp.reduce_snapshot(pl["meta"], pl["frames"], idx + 1)
    assert snap["phase"] == "block" and snap["block_idx"] == 1
    # seek 到 10s → freeze 之前的校准态（无更早相位边界 → idx=-1，零帧重建）
    idx0 = webapp.boundary_before(pl["frames"], pl["offsets_ms"], 10_000)
    assert idx0 == -1
    snap0 = webapp.reduce_snapshot(pl["meta"], pl["frames"], idx0 + 1)
    assert snap0["phase"] == "calibration" and snap0["k"] is None
