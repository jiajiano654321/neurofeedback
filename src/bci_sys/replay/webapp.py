"""IT4 Web 层（spec §6）：FastAPI app 工厂。

- live：POST /api/start 起引擎 daemon 线程；WS /ws 连接先发 snapshot 再桥接
  EventBus；单连接（第二连接 1013 Try Again Later）；WS 断开不中止引擎；关页不中止。
- replay：不起引擎、不写文件；GET /api/replay?file= 只允许启动时白名单解析出的
  单一 feedback.jsonl，返回 {meta, frames, offsets_ms, boundaries}，同一前端渲染；
  live 专有 WS/POST 在 replay 模式拒绝（1008/403）。
- 静态资源无外链；只服务 127.0.0.1（绑定在 CLI，见 feedback_cli）。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import queue
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .. import config, io_bids
from . import feedback_engine as eng

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

_CARD_MS = int(config.FEEDBACK_FREEZE_CARD_SEC * 1000)
_REST_MS = int(config.FEEDBACK_REST_SEC * 1000)
_WS_QUEUE_POLL_SEC = 0.5
# WS 桥接队列轮询间隔：有帧时 q.get 立即返回（零额外延迟），空队列时每
# _WS_QUEUE_POLL_SEC 回事件循环一次，使客户端断开与 uvicorn 优雅关闭
# 都能被感知；空转 CPU 开销可忽略，断连感知延迟上限即该值。


# ---------------- replay：jsonl → 同协议帧 + 墙钟偏移（只算不落盘） ----------------

def load_replay(path: str | Path) -> dict:
    """读一次 feedback.jsonl，拆 meta/frames，并给每帧算 1x 重放墙钟偏移（ms）。

    数据钟：seg/calib 段 0.2s；冻结卡 3s、每个 rest 30s 为纯墙钟（数据钟暂停）。
    offsets_ms 是播放节奏用的表现层量，绝不回写（A4）。
    """
    rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines()]
    if not rows or rows[0].get("type") != "meta":
        raise ValueError(f"非法 feedback.jsonl：首行不是 meta（{path}）")
    meta, frames = rows[0], rows[1:]
    offsets_ms: list[int] = []
    carry = 0
    for f in frames:
        t = f.get("type")
        if t == "calib_seg":
            wall = int((f["k"] + 1) * config.SEGMENT_SEC * 1000)
        elif t == "freeze":
            wall = int(f["t_sec"] * 1000)
        elif t == "phase" and f["phase"] == "block_start":
            wall = int(f["t1"] * 1000) + _CARD_MS + f["block_idx"] * _REST_MS
        elif t == "seg":
            wall = int(f["t1"] * 1000) + _CARD_MS + f["block_idx"] * _REST_MS
        elif t == "block_summary":
            wall = int(f["t1_end"] * 1000) + _CARD_MS + f["block_idx"] * _REST_MS
        elif t == "phase" and f["phase"] == "rest_start":
            wall = carry  # 紧贴上一块 summary 时刻
        elif t == "phase" and f["phase"] == "rest_end":
            wall = carry + _REST_MS
        else:  # final / aborted：停在上一帧时刻
            wall = carry
        offsets_ms.append(wall)
        carry = wall
    boundaries = [
        i for i, f in enumerate(frames)
        if f.get("type") in ("freeze", "final")
        or (f.get("type") == "phase" and f["phase"] in ("block_start", "rest_start"))
    ]
    return {"file": Path(path).name, "meta": meta, "frames": frames,
            "offsets_ms": offsets_ms, "boundaries": boundaries}


def reduce_snapshot(meta: dict, frames: list[dict], upto: int) -> dict:
    """从头回放 frames[:upto] 重建 snapshot（replay 首帧与 seek 后追平共用；
    与引擎 snapshot 同字段语义，replay 模式 status 由 final/aborted 定）。"""
    snap = {
        "type": "snapshot", "mode": "replay", "phase": "calibration",
        "block_idx": None, "seg_in_phase": 0, "k": None, "t1": None,
        "b_fp1": None, "b_fp2": None, "theta_thr": None,
        "params": meta["params"], "n_valid": 0, "n_hit": 0, "block_rates": [],
        "last_smoothed": None, "last_verdict": None,
        "status": None, "reason": None,
    }
    for f in frames[:upto]:
        t = f.get("type")
        if t in ("calib_seg", "calib_tick"):
            snap.update(phase="calibration", seg_in_phase=f["k"] + 1,
                        k=f["k"], t1=f["t1"])
        elif t == "freeze":
            snap.update(phase="freeze_card", seg_in_phase=0,
                        k=None, t1=f["t_sec"], b_fp1=f["b_fp1"],
                        b_fp2=f["b_fp2"], theta_thr=f["theta_thr_uV2"])
        elif t == "phase" and f["phase"] == "block_start":
            snap.update(phase="block", block_idx=f["block_idx"],
                        seg_in_phase=0, t1=f["t1"])
        elif t == "seg":
            snap.update(phase="block", block_idx=f["block_idx"],
                        seg_in_phase=f["seg_in_block"] + 1, k=f["k"], t1=f["t1"],
                        last_smoothed=f["smoothed_uV2"],
                        last_verdict={"valid": f["valid"], "reasons": f["reasons"],
                                      "fz_clean_p2p_uv": f["fz_clean_p2p_uv"],
                                      "eog_p2p_uv": f["eog_p2p_uv"]})
            # 口径与 FeedbackSession.record_segment 一致：valid 且功率有限才进 n_valid
            if f["valid"] and f["theta_pow_uV2"] is not None:
                snap["n_valid"] += 1
                if f["hit"]:
                    snap["n_hit"] += 1
        elif t == "block_summary":
            snap["block_rates"].append(f["hit_rate"])
        elif t == "phase" and f["phase"] == "rest_start":
            snap.update(phase="rest", seg_in_phase=0)
        elif t == "phase" and f["phase"] == "rest_end":
            snap["phase"] = "rest"
        elif t == "final":
            snap.update(phase="finished", status="ok",
                        n_valid=f["n_valid"], n_hit=f["n_hit"],
                        block_rates=list(f["block_rates"]))
        elif t == "aborted":
            snap.update(phase="aborted", status="aborted", reason=f["reason"],
                        k=f["k"], t1=f["t1"])
    return snap


def boundary_before(frames: list[dict], offsets_ms: list[int], wall_ms: int) -> int:
    """seek：wall_ms 处或之前最近的相位边界帧下标（freeze / block_start /
    rest_start / final；无更早边界时返回 -1，调用方从校准态起播）。"""
    idx = -1
    for i, f in enumerate(frames):
        if offsets_ms[i] > wall_ms:
            break
        if (f.get("type") in ("freeze", "final")
                or (f.get("type") == "phase" and f["phase"]
                    in ("block_start", "rest_start"))):
            idx = i
    return idx


def _default_runner_factory(subj, root, out_root):
    return eng.FeedbackRunner(subj=subj, root=root, out_root=out_root)


def create_app(
    *,
    mode: str = "live",
    subj: str = "001",
    data_root=None,
    out_root: str = "outputs",
    runner_factory=None,
    replay_file: str | Path | None = None,
) -> FastAPI:
    if mode not in ("live", "replay"):
        raise ValueError(f"mode 只能是 live/replay，收到 {mode!r}")
    if mode == "replay" and replay_file is None:
        raise ValueError("replay 模式必须给 replay_file")
    replay_resolved = Path(replay_file).resolve() if replay_file is not None else None
    # replay 在构造时读一次（CLI 已确认文件存在）；live 不预读任何数据
    replay_payload = load_replay(replay_resolved) if replay_resolved is not None else None
    runner_factory = runner_factory or _default_runner_factory

    app = FastAPI(title="bci-sys feedback-loop", docs_url=None, redoc_url=None)
    app.state.mode = mode
    app.state.runner = None

    def _static_run_info():
        # 确认页在引擎不存在时就要能回显 subj/数据文件/靶点（C7：参数先确认后启动）
        info = {
            "mode": mode, "subj": subj,
            "target": {"channel": config.TARGET.channel,
                       "band_hz": list(config.TARGET.band_hz),
                       "direction": config.TARGET.direction},
        }
        if mode == "replay":
            info.update(data_vhdr=replay_payload["meta"].get("data_vhdr"),
                        replay_file=replay_resolved.name)
        else:
            info["data_vhdr"] = io_bids.vhdr_path(subj, data_root).name  # 纯路径拼接，不读盘
        return info

    @app.get("/")
    def index():
        return FileResponse(_WEB_DIR / "index.html", media_type="text/html")

    @app.get("/app.js")
    def app_js():
        return FileResponse(_WEB_DIR / "app.js", media_type="text/javascript")

    @app.get("/api/run")
    def api_run():
        info = _static_run_info()
        r = app.state.runner
        if r is None:
            return {**info, "status": None, "reason": None, "run_dir": None}
        run_json_path = Path(r.run_dir) / "run.json"
        try:
            extra = json.loads(run_json_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            # OSError：文件尚不存在/权限等；JSONDecodeError：Windows write_text 非原子
            # （先截断再写），终态瞬间 GET 最密集时可能读到半截 JSON，不得 500。
            extra = {}
        if extra:
            # run.json 已落盘（终态帧 publish 前写入），直接以文件为准
            return {**info, "status": extra.get("status"),
                    "reason": extra.get("reason"),
                    "run_dir": r.run_dir.replace("\\", "/"), **extra}
        outcome = getattr(r, "outcome", None)
        if outcome is not None:
            return {**info, "status": outcome["status"],
                    "reason": outcome.get("reason"),
                    "run_dir": r.run_dir.replace("\\", "/")}
        return {**info, "status": None, "reason": None,
                "run_dir": r.run_dir.replace("\\", "/")}

    if mode == "live":
        @app.post("/api/start")
        def api_start():
            if app.state.runner is not None:
                raise HTTPException(409, "会话已存在（一次性，不可二次启动）")
            runner = runner_factory(subj=subj, root=data_root, out_root=out_root)
            app.state.runner = runner
            runner.start()
            return {"run_dir": runner.run_dir.replace("\\", "/")}

        @app.post("/api/abort")
        def api_abort(response: Response):
            runner = app.state.runner
            if runner is None:
                raise HTTPException(409, "尚无运行中的会话")
            if getattr(runner, "outcome", None) is not None:
                raise HTTPException(409, "会话已终态")
            runner.abort.set()
            response.status_code = 202
            return {"accepted": True}

        @app.websocket("/ws")
        async def ws_bridge(ws: WebSocket):
            # 单连接：第二连接 accept 后立刻以 1013 关闭。1013 是 RFC 6455
            # §7.4.2 注册码（Try Again Later），语义即“已有一个观看者，请稍后
            # 再试”；429 是 HTTP 状态码、不是合法 WebSocket 关闭码，真实
            # websockets 栈收到 429 会抛 ProtocolError 并在服务端刷 traceback。
            if getattr(app.state, "ws_taken", False):
                await ws.accept()
                await ws.close(code=1013)
                return
            app.state.ws_taken = True
            await ws.accept()

            async def _recv_sentinel():
                """感知客户端断开：只 receive 不处理消息（协议为服务端单向推流），
                连接关闭时以 WebSocketDisconnect 退出，供主循环检查。"""
                while True:
                    await ws.receive_text()

            recv_task = asyncio.create_task(_recv_sentinel())
            try:
                runner = app.state.runner
                # seq 与 snapshot 必须锁内一次读出（snapshot_pair），否则两次
                # 无锁读之间 publish 可能插入，发出 {seq:N+1, msg:snapshot_N}
                if runner is not None:
                    last_seq, snap = runner.bus.snapshot_pair()
                else:
                    last_seq, snap = 0, None
                if snap is not None:
                    await ws.send_json({"seq": last_seq, "msg": snap})
                else:
                    await ws.send_json({"seq": 0, "msg": eng.initial_snapshot()})
                while True:
                    if recv_task.done():
                        # recv_task 正常退出说明客户端已断开（WebSocketDisconnect
                        # 会被外层 except 捕获；任何其他异常也同样冒泡）
                        recv_task.result()
                        break
                    runner = app.state.runner  # start 可能发生在连接之后
                    if runner is None:
                        await asyncio.sleep(0.1)  # 引擎未启动：只等不占队列
                        continue
                    # 用超时轮询代替纯阻塞 q.get：终态后队列永久为空，
                    # 纯阻塞会钉死协程在 run_in_threadpool 的线程池 worker 里，
                    # 导致客户端断开与 uvicorn 优雅关闭都无法推进。
                    # 正常帧路径零额外延迟（q.get 有帧时立即返回）；
                    # 无帧时每 _WS_QUEUE_POLL_SEC 回事件循环一次，配合 recv_task 感知断开。
                    # queue.Empty 经 run_in_threadpool 从 worker 线程原样回传。
                    try:
                        env = await run_in_threadpool(
                            lambda: runner.bus.q.get(timeout=_WS_QUEUE_POLL_SEC))
                    except queue.Empty:
                        continue
                    # 丢弃 snapshot 追平点之前（含同 seq）的积压帧：页面晚开/
                    # 重连握手期队列里压的全是旧帧，转发会违反前端
                    # env.seq===lastSeq+1 契约，触发断档重连风暴（队列满丢最旧
                    # 只影响积压量，seq 仍单调，新帧 seq 必然 > last_seq）。
                    if env["seq"] <= last_seq:
                        continue
                    last_seq = env["seq"]
                    await ws.send_json(env)
                    # 终态帧送完后连接挂起等前端关闭；引擎线程已结束，不影响落盘
            except WebSocketDisconnect:
                pass
            finally:
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await recv_task
                app.state.ws_taken = False  # 断开不中止引擎，只释放连接名额

    else:  # replay
        @app.get("/api/replay")
        def api_replay(file: str = Query(..., min_length=1)):
            target = (replay_resolved.parent / file).resolve()
            if target != replay_resolved:
                # 白名单 = 启动参数解析出的单一文件；越界/穿越/换名全部 400
                raise HTTPException(400, "file 不在 replay 白名单内")
            # 构造时已读一次，端点直接复用内存快照（不复读、不解析）
            return replay_payload

        @app.post("/api/start")
        async def replay_start():
            raise HTTPException(403, "replay 模式为只读")

        @app.post("/api/abort")
        async def replay_abort():
            raise HTTPException(403, "replay 模式为只读")

        @app.websocket("/ws")
        async def ws_replay_reject(ws: WebSocket):
            await ws.accept()
            await ws.close(code=1008)  # 1008=policy violation：replay 无 live 流

    return app
