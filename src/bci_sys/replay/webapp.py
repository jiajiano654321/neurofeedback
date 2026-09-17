"""IT4 Web 层（spec §6）：FastAPI app 工厂。

- live：POST /api/start 起引擎 daemon 线程；WS /ws 连接先发 snapshot 再桥接
  EventBus；单连接（第二连接 429）；WS 断开不中止引擎；关页不中止。
- replay：不起引擎、不写文件；GET /api/replay?file= 只允许启动时白名单解析出的
  单一 feedback.jsonl，返回 {meta, frames, offsets_ms, boundaries}，同一前端渲染；
  live 专有 WS/POST 在 replay 模式拒绝（1008/403）。
- 静态资源无外链；只服务 127.0.0.1（绑定在 CLI，见 feedback_cli）。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .. import config, io_bids
from . import feedback_engine as eng

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


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
            wall = int(f["t1"] * 1000) + 3000 + f["block_idx"] * 30_000
        elif t == "seg":
            wall = int(f["t1"] * 1000) + 3000 + f["block_idx"] * 30_000
        elif t == "block_summary":
            wall = int(f["t1_end"] * 1000) + 3000 + f["block_idx"] * 30_000
        elif t == "phase" and f["phase"] == "rest_start":
            wall = carry  # 紧贴上一块 summary 时刻
        elif t == "phase" and f["phase"] == "rest_end":
            wall = carry + 30_000
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
            if f["valid"]:
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
        outcome = getattr(r, "outcome", None)
        if outcome is None:
            return {**info, "status": None, "reason": None,
                    "run_dir": r.run_dir.replace("\\", "/")}
        # 终态后回读 run.json：数据/墙钟时长只从这里进浏览器（不进 WS、不进 jsonl）
        run_json_path = Path(r.run_dir) / "run.json"
        extra = {}
        try:
            extra = json.loads(run_json_path.read_text("utf-8"))
        except OSError:
            pass
        return {**info, "status": outcome["status"],
                "reason": outcome.get("reason"),
                "run_dir": r.run_dir.replace("\\", "/"), **extra}

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
            import asyncio
            from starlette.concurrency import run_in_threadpool

            # 单连接：第二连接 accept 后立刻以 429 关闭（跨 Starlette 版本可移植）
            if getattr(app.state, "ws_taken", False):
                await ws.accept()
                await ws.close(code=429)
                return
            app.state.ws_taken = True
            await ws.accept()
            try:
                runner = app.state.runner
                if runner is not None and runner.bus.snapshot is not None:
                    await ws.send_json({"seq": runner.bus.seq,
                                        "msg": runner.bus.snapshot})
                else:
                    await ws.send_json({"seq": 0, "msg": eng.initial_snapshot()})
                while True:
                    runner = app.state.runner  # start 可能发生在连接之后
                    if runner is None:
                        await asyncio.sleep(0.1)  # 引擎未启动：只等不占队列
                        continue
                    # q.get 是阻塞调用，经线程池桥接（单连接，最多占一个 worker）
                    env = await run_in_threadpool(runner.bus.q.get)
                    await ws.send_json(env)
                    # 终态帧送完后连接挂起等前端关闭；引擎线程已结束，不影响落盘
            except WebSocketDisconnect:
                pass
            finally:
                app.state.ws_taken = False  # 断开不中止引擎，只释放连接名额

    else:  # replay
        @app.get("/api/replay")
        def api_replay(file: str = Query(..., min_length=1)):
            target = (replay_resolved.parent / file).resolve()
            if target != replay_resolved:
                # 白名单 = 启动参数解析出的单一文件；越界/穿越/换名全部 400
                raise HTTPException(400, "file 不在 replay 白名单内")
            try:
                return load_replay(target)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(400, f"replay 文件不可读：{exc}")

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
