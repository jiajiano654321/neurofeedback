"""IT4 反馈会话引擎（后台 daemon 线程内的同步实时循环；spec §3–§5、§7）。

preflight → 读 Fz/FP1/FP2 → 三路 1–30Hz 因果带通 → S3 校准
（冻结 b 与 θ 阈值）→ 冻结卡 3s → 6×(120s block + 30s rest) →
feedback.jsonl / run.json。事件帧经 EventBus（有界 256、丢最旧、带 seq）
交给 Web 层；引擎永不阻塞在 UI 上。

A4：jsonl 只含数据侧量，两次运行字节一致；墙钟/线程/队列量只进 run.json。
C5：无效段 smoothed/hit=null、计数不动；数据只标记不删除。
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config, gitmeta, io_bids
from . import causal_filter, feedback, qc, theta_metric

SCHEMA_VERSION = "1.0"
FZ = config.TARGET.channel
EOGS = config.EOG_PROXY_CHANNELS
ALL_CH = (FZ, *EOGS)
SEG = config.SEGMENT_SAMPLES
CALIB_SEGS = int(round(config.QC_CALIBRATION_SEC / config.SEGMENT_SEC))
BLOCK_SEGS = int(round(config.FEEDBACK_BLOCK_SEC / config.SEGMENT_SEC))


def _finite_or_none(v):
    """非有限值（NaN/Inf）落盘为 null：裸 NaN/Infinity 不是合法 JSON token。"""
    if v is None:
        return None
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def build_params() -> dict:
    """meta.params 与初始 snapshot.params 共用同一份冻结常量（C2），字段顺序即落盘顺序。"""
    return {
        "sfreq": config.SFREQ,
        "segment_sec": config.SEGMENT_SEC,
        "qc_calibration_sec": config.QC_CALIBRATION_SEC,
        "qc_filter_warmup_sec": config.QC_FILTER_WARMUP_SEC,
        "l1_p2p_uv": config.QC_L1_P2P_UV,
        "l3_eog_p2p_uv": config.QC_L3_EOG_P2P_UV,
        "theta_band_hz": list(config.THETA_BAND_HZ),
        "theta_smooth_segs": config.THETA_SMOOTH_SEGS,
        "theta_quantile": config.THETA_QUANTILE,
        "theta_target_rate": config.THETA_TARGET_RATE,
        "theta_min_calib_segs": config.THETA_MIN_CALIB_SEGS,
        "n_blocks": config.FEEDBACK_N_BLOCKS,
        "block_sec": config.FEEDBACK_BLOCK_SEC,
        "rest_sec": config.FEEDBACK_REST_SEC,
        "freeze_card_sec": config.FEEDBACK_FREEZE_CARD_SEC,
        "session_wall_cap_sec": config.SESSION_WALL_CAP_SEC,
    }


def initial_snapshot() -> dict:
    """WS 首帧 snapshot（live）；引擎每发一帧用最新值整体重建。"""
    return {
        "type": "snapshot",
        "mode": "live",
        "phase": "calibration",
        "block_idx": None,
        "seg_in_phase": 0,
        "k": None,
        "t1": None,
        "b_fp1": None,
        "b_fp2": None,
        "theta_thr": None,
        "params": build_params(),
        "n_valid": 0,
        "n_hit": 0,
        "block_rates": [],
        "last_smoothed": None,
        "last_verdict": None,
        "status": None,
        "reason": None,
    }


class EventBus:
    """引擎→Web 的有界单向通道（线程安全）。

    put_nowait 永不阻塞引擎：满则丢最旧帧并累加 dropped；每帧带单调 seq，
    客户端断档后凭 snapshot 重同步。snapshot 由引擎每帧重建、随帧携带。
    """

    def __init__(self, maxsize: int = 256):
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.seq = 0
        self.snapshot: dict | None = None
        self._lock = threading.Lock()

    def publish(self, msg: dict, snapshot: dict) -> None:
        with self._lock:
            self.seq += 1
            snap = dict(snapshot)
            env = {"seq": self.seq, "msg": msg}
            try:
                self.q.put_nowait(env)
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.dropped += 1
                try:
                    self.q.put_nowait(env)
                except queue.Full:
                    pass
            snap["seq"] = self.seq
            snap["dropped"] = self.dropped
            self.snapshot = snap


def _interruptible_wait(seconds: float, abort_event, sleep_fn, step: float = 0.2) -> bool:
    """片长 ≤step 的可中止等待（中止响应 ≤200ms，spec §3）。True=走完，False=被中止。"""
    slept = 0.0
    while slept + 1e-9 < seconds:
        if abort_event is not None and abort_event.is_set():
            return False
        dt = min(step, seconds - slept)
        sleep_fn(dt)
        slept += dt
    return not (abort_event is not None and abort_event.is_set())


def _compute_freeze(judge: qc.QcJudge, cal_buf: dict) -> dict:
    """spec §4.2 步骤 1–6：b 取自 QcJudge 同次冻结；重建 clean、重算 600 段
    有效性（同一 L1/L3 规则）、一次性 4–8Hz 取 zf、P20 阈值。"""
    cal = judge.calib
    b1, b2 = cal["b_fp1"], cal["b_fp2"]
    n = judge.calib_samples
    arrs = {c: np.concatenate(cal_buf[c])[:n] for c in ALL_CH}
    cal_valid = []
    for j in range(CALIB_SEGS):
        sl = slice(j * SEG, j * SEG + SEG)
        cal_valid.append(qc.segment_verdict(
            arrs[FZ][sl], arrs[EOGS[0]][sl], arrs[EOGS[1]][sl],
            b1, b2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV)["valid"])
    clean = arrs[FZ] - b1 * arrs[EOGS[0]] - b2 * arrs[EOGS[1]]
    out = theta_metric.calibrate_theta(clean, np.asarray(cal_valid, dtype=bool))
    out["b_fp1"] = b1
    out["b_fp2"] = b2
    out["cal_valid"] = cal_valid
    out["n_fit_samples"] = cal["n_fit_samples"]
    out["fit_window_sec"] = list(cal["fit_window_sec"])
    return out


class _Stop(Exception):
    """引擎内部控制流：终态行已写，跳出 block 循环。"""


def run_feedback(
    *,
    subj: str,
    root=None,
    out_root: str | Path = "outputs",
    run_id: str | None = None,
    abort_event=None,
    bus: EventBus | None = None,
    sleep_fn=time.sleep,
    clock=time.perf_counter,
    chans: dict | None = None,
) -> dict:
    """跑一次 live 反馈会话。返回 {status, reason, run_dir}。

    preflight 失败直接抛异常（不建目录，CLI 映射 rc=2）。
    chans=None 时自己 preflight+读盘；CLI 在 uvicorn 启动前已完成这两步，
    会把读好的三路数组作为 chans 传入（保证服务器起来后不再可能因数据问题 rc=2）。
    """
    if chans is None:
        io_bids.preflight(subj, root)  # 硬停，不写任何文件
        chans = io_bids.load_channels_uV(subj, list(ALL_CH), root)
    n_avail = chans[FZ].size // SEG

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    print(
        f"[feedback-loop] sub-{subj} | S3 校准 {config.QC_CALIBRATION_SEC:g}s | "
        f"S4 {config.FEEDBACK_N_BLOCKS}×{config.FEEDBACK_BLOCK_SEC:g}s "
        f"rest {config.FEEDBACK_REST_SEC:g}s | θ {config.THETA_BAND_HZ[0]:g}–"
        f"{config.THETA_BAND_HZ[1]:g}Hz W={config.THETA_SMOOTH_SEGS} "
        f"P{config.THETA_QUANTILE*100:g} direction={config.TARGET.direction}"
    )
    run_dir = Path(out_root) / "feedback" / f"sub-{subj}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    wall0 = clock()
    wall_iso = datetime.now().isoformat(timespec="seconds")
    bus = bus or EventBus()
    band = causal_filter.CausalBandpass(ALL_CH)
    judge = qc.QcJudge()
    session = feedback.FeedbackSession()
    run_smoother = theta_metric.ValidOnlySmoother()
    theta_stream = None
    decider = None
    cal_buf: dict | None = {c: [] for c in ALL_CH}

    params = build_params()
    snap = initial_snapshot()
    snap["params"] = params
    last_k, last_t1 = 0, 0.0
    n_cal_rows, n_run_rows = 0, 0
    pause_total = 0.0
    terminal_row: dict | None = None

    vhdr = io_bids.vhdr_path(subj, root).name
    jsonl_path = run_dir / "feedback.jsonl"

    def wall_elapsed() -> float:
        return clock() - wall0

    def wait_data(k: int) -> None:
        """数据钟 1x 节拍；rest/冻结卡的墙钟通过 pause_total 平移 deadline。"""
        target = wall0 + (k + 1) * config.SEGMENT_SEC + pause_total
        dt = target - clock()
        if dt > 0:
            sleep_fn(dt)

    def aborted() -> bool:
        return abort_event is not None and abort_event.is_set()

    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "subj": subj,
            "data_vhdr": vhdr,
            "git_commit": gitmeta.code_version(),
            "target": {
                "channel": config.TARGET.channel,
                "band_hz": list(config.TARGET.band_hz),
                "direction": config.TARGET.direction,
            },
            "params": params,
        }, ensure_ascii=False) + "\n")
        f.flush()

        try:
            # ---------- S3 校准（k=0…599） ----------
            k = 0
            while k < CALIB_SEGS:
                if k >= n_avail:
                    terminal_row = session.terminate("incomplete_data", k=last_k, t1=last_t1)
                    break
                if aborted():
                    terminal_row = session.terminate("user_abort", k=last_k, t1=last_t1)
                    break
                s, e = k * SEG, k * SEG + SEG
                t1 = round(e / config.SFREQ, 6)
                filtered = band.push_block({c: chans[c][s:e] for c in ALL_CH})
                try:
                    verdict = judge.push(t1, filtered)
                except ValueError as exc:  # EOG 拟合失败：冻结失败硬停
                    terminal_row = session.terminate(
                        "freeze_failed", k=k, t1=t1, error=str(exc))
                    break
                if cal_buf is not None:
                    for c in ALL_CH:
                        cal_buf[c].append(filtered[c])
                last_k, last_t1 = k, t1
                bus.publish({"type": "calib_tick", "k": k, "t1": t1},
                            {**snap, "phase": "calibration", "seg_in_phase": k + 1,
                             "k": k, "t1": t1})
                wait_data(k)
                cap = session.wall_cap_terminate(wall_elapsed(), k=k, t1=t1)
                if cap is not None:
                    terminal_row = cap
                    break
                k += 1

                if verdict.get("calib") is not None:
                    try:
                        fr = _compute_freeze(judge, cal_buf)
                        # note_freeze 二次守门（成熟段 <50 / 阈值非有限）
                        session.note_freeze(fr)
                    except theta_metric.FreezeFailed as exc:
                        terminal_row = session.terminate(
                            "freeze_failed", k=k - 1, t1=t1, error=str(exc))
                        break
                    b1, b2 = fr["b_fp1"], fr["b_fp2"]
                    for j in range(CALIB_SEGS):
                        f.write(json.dumps({
                            "type": "calib_seg",
                            "k": j,
                            "t1": round((j + 1) * SEG / config.SFREQ, 6),
                            "theta_pow_uV2": _finite_or_none(float(fr["cal_powers"][j])),
                            "valid": bool(fr["cal_valid"][j]),
                            "smoothed_uV2": _finite_or_none(fr["cal_smoothed"][j]),
                        }, ensure_ascii=False) + "\n")
                    n_cal_rows = CALIB_SEGS
                    freeze_row = {
                        "type": "freeze",
                        "t_sec": round(config.QC_CALIBRATION_SEC, 6),
                        "b_fp1": b1,
                        "b_fp2": b2,
                        "theta_thr_uV2": fr["theta_thr_uV2"],
                        "smooth_win_seg": config.THETA_SMOOTH_SEGS,
                        "quantile": config.THETA_QUANTILE,
                        "target_rate": config.THETA_TARGET_RATE,
                        "direction": config.TARGET.direction,
                        "n_thr_segs": fr["n_thr_segs"],
                        "calib_hit_rate": fr["calib_hit_rate"],
                        "n_fit_samples": fr["n_fit_samples"],
                        "fit_window_sec": fr["fit_window_sec"],
                    }
                    f.write(json.dumps(freeze_row, ensure_ascii=False) + "\n")
                    f.flush()
                    cal_buf = None  # 释放 ~0.72MB（720KB）校准缓冲
                    theta_stream = theta_metric.ThetaStream(zf=fr["zf"])
                    decider = theta_metric.HitDecider(
                        fr["theta_thr_uV2"], config.TARGET.direction)
                    snap.update({
                        "phase": "freeze_card", "seg_in_phase": 0,
                        "k": k - 1, "t1": config.QC_CALIBRATION_SEC,
                        "b_fp1": b1, "b_fp2": b2,
                        "theta_thr": fr["theta_thr_uV2"],
                    })
                    bus.publish(freeze_row, snap)
                    print(f"[冻结] b_{EOGS[0]}={b1:.4f} b_{EOGS[1]}={b2:.4f} "
                          f"θ_thr={fr['theta_thr_uV2']:.2f}µV² "
                          f"n={fr['n_thr_segs']} hit_rate={fr['calib_hit_rate']:.3f}")
                    if not _interruptible_wait(
                        config.FEEDBACK_FREEZE_CARD_SEC, abort_event, sleep_fn):
                        terminal_row = session.terminate(
                            "user_abort", k=k - 1, t1=config.QC_CALIBRATION_SEC)
                        break
                    pause_total += config.FEEDBACK_FREEZE_CARD_SEC
                    cap = session.wall_cap_terminate(wall_elapsed(), k=k - 1,
                                                     t1=config.QC_CALIBRATION_SEC)
                    if cap is not None:
                        terminal_row = cap
                        break
                    bs_row = session.begin_block(config.QC_CALIBRATION_SEC)
                    f.write(json.dumps(bs_row, ensure_ascii=False) + "\n")
                    f.flush()
                    snap.update({"phase": "block", "block_idx": 0, "seg_in_phase": 0,
                                 "t1": config.QC_CALIBRATION_SEC})
                    bus.publish(bs_row, snap)
                    break

            # ---------- S4：6×(block + rest) ----------
            if terminal_row is None and theta_stream is not None:
                try:
                    for bi in range(config.FEEDBACK_N_BLOCKS):
                        if bi > 0:
                            rs = session.begin_rest(config.FEEDBACK_REST_SEC)
                            f.write(json.dumps(rs, ensure_ascii=False) + "\n")
                            f.flush()
                            snap.update({"phase": "rest", "seg_in_phase": 0})
                            bus.publish(rs, snap)
                            if not _interruptible_wait(
                                config.FEEDBACK_REST_SEC, abort_event, sleep_fn):
                                terminal_row = session.terminate(
                                    "user_abort", k=last_k, t1=last_t1)
                                raise _Stop
                            pause_total += config.FEEDBACK_REST_SEC
                            cap = session.wall_cap_terminate(
                                wall_elapsed(), k=last_k, t1=last_t1)
                            if cap is not None:
                                terminal_row = cap
                                raise _Stop
                            re_ = session.end_rest(config.FEEDBACK_REST_SEC)
                            f.write(json.dumps(re_, ensure_ascii=False) + "\n")
                            f.flush()
                            bus.publish(re_, snap)
                            t1b = round(config.QC_CALIBRATION_SEC + bi * config.FEEDBACK_BLOCK_SEC, 6)
                            bs = session.begin_block(t1b)
                            f.write(json.dumps(bs, ensure_ascii=False) + "\n")
                            f.flush()
                            snap.update({"phase": "block", "block_idx": bi,
                                         "seg_in_phase": 0, "t1": t1b})
                            bus.publish(bs, snap)

                        for j in range(BLOCK_SEGS):
                            k = CALIB_SEGS + bi * BLOCK_SEGS + j
                            if k >= n_avail:
                                terminal_row = session.terminate(
                                    "incomplete_data", k=last_k, t1=last_t1)
                                raise _Stop
                            if aborted():
                                terminal_row = session.terminate(
                                    "user_abort", k=last_k, t1=last_t1)
                                raise _Stop
                            s0, s1 = k * SEG, k * SEG + SEG
                            t1 = round(s1 / config.SFREQ, 6)
                            filtered = band.push_block(
                                {c: chans[c][s0:s1] for c in ALL_CH})
                            v = judge.push(t1, filtered)
                            clean50 = (filtered[FZ] - b1 * filtered[EOGS[0]]
                                       - b2 * filtered[EOGS[1]])
                            _, powv = theta_stream.push_segment(clean50)
                            finite = math.isfinite(powv)
                            sm = run_smoother.update(powv, bool(v["valid"]))
                            hit = None if (not v["valid"] or sm is None) \
                                else bool(decider.hit(sm))
                            session.record_segment(
                                valid=v["valid"], power_finite=finite, hit=hit)
                            last_k, last_t1 = k, t1
                            seg_row = {
                                "type": "seg",
                                "k": k,
                                "t1": t1,
                                "block_idx": bi,
                                "seg_in_block": j,
                                "theta_pow_uV2": _finite_or_none(powv),
                                "smoothed_uV2": _finite_or_none(sm),
                                "hit": hit,
                                "fz_clean_p2p_uv": _finite_or_none(v["fz_clean_p2p"]),
                                "eog_p2p_uv": _finite_or_none(v["eog_p2p"]),
                                "valid": v["valid"],
                                "reasons": list(v["reasons"]),
                            }
                            f.write(json.dumps(seg_row, ensure_ascii=False) + "\n")
                            f.flush()
                            n_run_rows += 1  # 计数紧跟落盘、先于等待（Ctrl-C 落点一致性）
                            snap.update({
                                "phase": "block", "block_idx": bi,
                                "seg_in_phase": j + 1, "k": k, "t1": t1,
                                "n_valid": session.total.n_valid,
                                "n_hit": session.total.n_hit,
                                "block_rates": list(session.block_rates),
                                "last_smoothed": _finite_or_none(sm),
                                "last_verdict": {
                                    "valid": v["valid"],
                                    "reasons": list(v["reasons"]),
                                    "fz_clean_p2p_uv": _finite_or_none(v["fz_clean_p2p"]),
                                    "eog_p2p_uv": _finite_or_none(v["eog_p2p"]),
                                },
                            })
                            bus.publish(seg_row, snap)
                            wait_data(k)
                            cap = session.wall_cap_terminate(wall_elapsed(), k=k, t1=t1)
                            if cap is not None:
                                terminal_row = cap
                                raise _Stop

                        t1_end = round(
                            config.QC_CALIBRATION_SEC + (bi + 1) * config.FEEDBACK_BLOCK_SEC, 6)
                        summ = session.end_block_if_due(t1_end)
                        f.write(json.dumps(summ, ensure_ascii=False) + "\n")
                        f.flush()
                        snap.update({"block_rates": list(session.block_rates)})
                        bus.publish(summ, snap)
                        print(f"[block {bi + 1}/6] hit_rate={summ['hit_rate']:.3f} "
                              f"valid={summ['n_valid']} invalid={summ['n_invalid']}")

                    final_row = session.finish()
                    f.write(json.dumps(final_row, ensure_ascii=False) + "\n")
                    f.flush()
                    snap.update({"phase": "finished", "status": "ok",
                                 "n_valid": session.total.n_valid,
                                 "n_hit": session.total.n_hit,
                                 "block_rates": list(session.block_rates)})
                    bus.publish(final_row, snap)
                except _Stop:
                    pass
        except Exception as exc:  # 未预期异常：engine_error 硬停，traceback 只进终端
            traceback.print_exc()
            terminal_row = session.terminate(
                "engine_error", k=last_k, t1=last_t1, error=type(exc).__name__)

        if terminal_row is not None:
            f.write(json.dumps(terminal_row, ensure_ascii=False) + "\n")
            f.flush()
            snap.update({"phase": "aborted", "status": "aborted",
                         "reason": terminal_row["reason"],
                         "k": terminal_row["k"], "t1": terminal_row["t1"]})
            bus.publish(terminal_row, snap)

    wall = wall_elapsed()
    run_json = {
        "subj": subj,
        "status": session.status,
        "mode": "live",
        "wall_started_iso": wall_iso,
        "wall_elapsed_s": round(wall, 3),
        "data_elapsed_s": round(last_t1, 3),
        "rate_ratio": round(last_t1 / wall, 4) if wall > 0 else 1.0,
        "n_calib_seg": n_cal_rows,
        "n_running_seg": n_run_rows,
        "block_rates": list(session.block_rates),
        "git_commit": gitmeta.code_version(),
    }
    if session.reason is not None:
        run_json["reason"] = session.reason
    (run_dir / "run.json").write_text(
        json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": session.status, "reason": session.reason, "run_dir": str(run_dir)}


class FeedbackRunner:
    """Web 层持有的引擎线程壳：POST /api/start 建一个，daemon 线程。"""

    def __init__(self, *, subj: str, root=None, out_root="outputs",
                 run_id: str | None = None, chans: dict | None = None):
        self.subj = subj
        self.root = root
        self.chans = chans
        self.out_root = out_root
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = str(
            Path(out_root) / "feedback" / f"sub-{subj}" / self.run_id)
        self.bus = EventBus()
        self.bus.snapshot = initial_snapshot()
        self.abort = threading.Event()
        self.outcome: dict | None = None
        self._sleep_fn = None
        self._clock = None
        self._t: threading.Thread | None = None

    def start(self, *, sleep_fn=None, clock=None) -> None:
        # 只在显式传参时覆盖：CLI 测试可能已在构造后预置快放时钟，
        # webapp 以无参 start() 调用，不能把预置值冲掉
        if sleep_fn is not None:
            self._sleep_fn = sleep_fn
        if clock is not None:
            self._clock = clock
        self._t = threading.Thread(target=self._target, name="feedback-engine", daemon=True)
        self._t.start()

    def _target(self) -> None:
        kwargs = {}
        if self._sleep_fn is not None:
            kwargs["sleep_fn"] = self._sleep_fn
        if self._clock is not None:
            kwargs["clock"] = self._clock
        try:
            self.outcome = run_feedback(
                subj=self.subj, root=self.root, out_root=self.out_root,
                run_id=self.run_id, abort_event=self.abort, bus=self.bus,
                chans=self.chans, **kwargs)
        except Exception:
            traceback.print_exc()
            self.outcome = {"status": "aborted", "reason": "engine_error",
                            "run_dir": self.run_dir}

    def join(self, timeout=None) -> None:
        if self._t is not None:
            self._t.join(timeout)

    @property
    def alive(self) -> bool:
        return self._t is not None and self._t.is_alive()
