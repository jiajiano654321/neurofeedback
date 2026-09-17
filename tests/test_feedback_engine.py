"""IT4 引擎：schema/顺序、A4 字节一致、abort/incomplete/freeze_failed/engine_error、
帧与 jsonl 同行一致、独立重算、内存上限、LF/非有限守卫。"""
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bci_sys import config, io_bids
from bci_sys.replay import feedback_engine as eng
from bci_sys.replay import theta_metric as tm, qc, causal_filter


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class GateClock:
    """t 进入 (lo, hi) 墙钟窗后阻塞，直到 abort 置位（测中止落点）。"""
    def __init__(self, abort, lo, hi=1e9):
        self.t = 0.0
        self.abort = abort
        self.lo, self.hi = lo, hi

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if self.lo < self.t < self.hi:
            # 正常毫秒级返回（abort 已 set 或 setter 很快置位）；30s 留 CI 高负载余量
            self.abort.wait(30.0)


def _rows(rd: Path):
    return [json.loads(x) for x in (rd / "feedback.jsonl").read_text("utf-8").splitlines()]


def _run(data_root, tmp, rid="fixed", **kw):
    clk = FakeClock()
    out = eng.run_feedback(
        subj="001", root=data_root, out_root=tmp, run_id=rid,
        sleep_fn=clk.sleep, clock=clk.now, **kw)
    rd = tmp / "feedback" / "sub-001" / rid
    return out, rd, clk


def test_full_run_schema_order_and_counts(data_root, tmp_path):
    out, rd, clk = _run(data_root, tmp_path)
    assert out["status"] == "ok" and out["reason"] is None
    rows = _rows(rd)
    kinds = [r["type"] for r in rows]
    assert kinds.count("meta") == 1 and kinds[0] == "meta"
    assert kinds.count("calib_seg") == 600
    assert kinds.count("freeze") == 1
    assert kinds.count("seg") == 3600
    assert kinds.count("block_summary") == 6
    assert kinds.count("final") == 1 and kinds[-1] == "final"
    phases = [r for r in rows if r["type"] == "phase"]
    assert sum(r["phase"] == "block_start" for r in phases) == 6
    assert sum(r["phase"] == "rest_start" for r in phases) == 5
    assert sum(r["phase"] == "rest_end" for r in phases) == 5
    assert len(phases) == 16
    # calib_seg 必须整体在 freeze 之前；freeze 后立刻是 block_start 0
    fi = kinds.index("freeze")
    assert kinds[1:fi] == ["calib_seg"] * 600
    assert rows[fi + 1] == {"type": "phase", "phase": "block_start",
                            "block_idx": 0, "t1": 120.0}
    # 每块：600 seg 后一条 summary；rest 夹在块间且只带 rest_idx/rest_sec
    si = fi + 2
    for b in range(6):
        assert all(r["block_idx"] == b for r in rows[si:si + 600])
        assert [r["seg_in_block"] for r in rows[si:si + 600]] == list(range(600))
        sm = rows[si + 600]
        assert sm["type"] == "block_summary" and sm["block_idx"] == b
        assert sm["t1_start"] == 120.0 + 120 * b and sm["t1_end"] == 240.0 + 120 * b
        assert sm["n_valid"] + sm["n_invalid"] == 600
        assert sm["hit_rate"] == (round(sm["n_hit"] / sm["n_valid"], 6) if sm["n_valid"] else 0.0)
        si += 601
        if b < 5:
            rs, re_ = rows[si], rows[si + 1]
            assert rs["phase"] == "rest_start" and re_["phase"] == "rest_end"
            assert rs["rest_idx"] == b and re_["rest_idx"] == b
            assert rs["rest_sec"] == 30.0 and re_["rest_sec"] == 30.0
            assert "wall" not in rs and "t1" not in rs
            assert rows[si + 2]["phase"] == "block_start" and rows[si + 2]["block_idx"] == b + 1
            si += 3


def test_meta_freeze_seg_row_field_sets(data_root, tmp_path):
    _, rd, _ = _run(data_root, tmp_path)
    rows = _rows(rd)
    meta = rows[0]
    assert meta["schema_version"] == "1.0"
    assert meta["data_vhdr"] == "sub-001_task-nback_eeg.vhdr"
    assert meta["target"] == {"channel": "Fz", "band_hz": [4.0, 8.0], "direction": "up"}
    assert set(meta["params"]) == {
        "sfreq", "segment_sec", "qc_calibration_sec", "qc_filter_warmup_sec",
        "l1_p2p_uv", "l3_eog_p2p_uv", "theta_band_hz", "theta_smooth_segs",
        "theta_quantile", "theta_target_rate", "theta_min_calib_segs",
        "n_blocks", "block_sec", "rest_sec", "freeze_card_sec", "session_wall_cap_sec"}
    assert meta["params"]["theta_smooth_segs"] == 5
    fz = rows[601]
    assert set(fz) == {"type", "t_sec", "b_fp1", "b_fp2", "theta_thr_uV2",
                       "smooth_win_seg", "quantile", "target_rate", "direction",
                       "n_thr_segs", "calib_hit_rate", "n_fit_samples", "fit_window_sec"}
    assert fz["t_sec"] == 120.0 and fz["n_fit_samples"] == 27500
    assert fz["fit_window_sec"] == [10.0, 120.0] and fz["direction"] == "up"
    assert fz["quantile"] == 0.20 and fz["target_rate"] == 0.80
    assert 0.1 <= fz["b_fp1"] <= 0.7 and 0.1 <= fz["b_fp2"] <= 0.7
    assert fz["n_thr_segs"] >= 450 and math.isfinite(fz["theta_thr_uV2"])
    seg = next(r for r in rows if r["type"] == "seg")
    assert set(seg) == {"type", "k", "t1", "block_idx", "seg_in_block",
                        "theta_pow_uV2", "smoothed_uV2", "hit",
                        "fz_clean_p2p_uv", "eog_p2p_uv", "valid", "reasons"}
    assert seg["k"] == 600 and seg["t1"] == 120.2
    # 实测无效段分布（只依赖 QC，与 θ 平滑无关；spec §2.6）
    summaries = [r for r in rows if r["type"] == "block_summary"]
    assert [s["n_invalid"] for s in summaries] == [43, 41, 57, 71, 85, 108]
    for s in summaries:
        assert 0.5 < s["hit_rate"] < 0.99  # 宽结构断言；具体数值进 evidence
    final = rows[-1]
    assert final["status"] == "ok" and len(final["block_rates"]) == 6
    assert final["n_valid"] == sum(s["n_valid"] for s in summaries)
    assert final["n_hit"] == sum(s["n_hit"] for s in summaries)


def test_run_json_matches_rows_and_wall_separated(data_root, tmp_path):
    _, rd, clk = _run(data_root, tmp_path)
    run = json.loads((rd / "run.json").read_text("utf-8"))
    rows = _rows(rd)
    assert run["status"] == "ok" and run["mode"] == "live"
    assert run["n_calib_seg"] == 600 and run["n_running_seg"] == 3600
    assert run["data_elapsed_s"] == 840.0
    assert run["wall_elapsed_s"] > run["data_elapsed_s"]  # 含 150s rest + 3s 卡
    assert 0.8 < run["rate_ratio"] < 0.9
    assert len(run["block_rates"]) == 6 and "git_commit" in run
    raw = (rd / "feedback.jsonl").read_bytes()
    assert b"\r" not in raw  # LF 锁定
    # 墙钟/线程/队列量只进 run.json，绝不进 jsonl（A4）。
    # 注意不能按子串 b"wall" 扫——冻结参数名 session_wall_cap_sec 合法含 "wall"。
    forbidden = {"wall_elapsed_s", "wall_started_iso", "rate_ratio", "seq", "dropped"}
    assert all(forbidden.isdisjoint(r) for r in rows)
    assert not any(k.endswith("_iso") for r in rows for k in r)


def test_a4_byte_identical_two_runs(data_root, tmp_path):
    bs = []
    for rid in ("a", "b"):
        clk = FakeClock()
        eng.run_feedback(subj="001", root=data_root, out_root=tmp_path / rid,
                         run_id="fixed", sleep_fn=clk.sleep, clock=clk.now)
        bs.append((tmp_path / rid / "feedback" / "sub-001" / "fixed" /
                   "feedback.jsonl").read_bytes())
    assert bs[0] == bs[1]
    assert hashlib.sha256(bs[0]).hexdigest() == hashlib.sha256(bs[1]).hexdigest()


def test_bus_frames_match_jsonl_rows_and_ticks_live_only(data_root, tmp_path):
    clk = FakeClock()
    bus = eng.EventBus(maxsize=100000)
    eng.run_feedback(subj="001", root=data_root, out_root=tmp_path, run_id="fixed",
                     sleep_fn=clk.sleep, clock=clk.now, bus=bus)
    frames = []
    while True:
        try:
            frames.append(bus.q.get_nowait())
        except Exception:
            break
    assert [f["msg"]["type"] for f in frames[:600]] == ["calib_tick"] * 600
    assert frames[0]["msg"] == {"type": "calib_tick", "k": 0, "t1": 0.2}
    assert all(f["seq"] == i + 1 for i, f in enumerate(frames))  # seq 连续
    rows = _rows(tmp_path / "feedback" / "sub-001" / "fixed")
    # meta 与 600 calib_seg 不走 live WS；其余帧 msg 与 jsonl 同行逐字相等
    persisted = rows[601:]
    assert [f["msg"] for f in frames[600:]] == persisted


def test_abort_preset_before_start(data_root, tmp_path):
    ev = threading.Event()
    ev.set()
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="a", sleep_fn=clk.sleep, clock=clk.now, abort_event=ev)
    assert out["status"] == "aborted" and out["reason"] == "user_abort"
    rd = tmp_path / "feedback" / "sub-001" / "a"
    rows = _rows(rd)
    assert rows[-1]["reason"] == "user_abort"
    assert not any(r["type"] in ("calib_seg", "freeze", "seg") for r in rows)
    run = json.loads((rd / "run.json").read_text("utf-8"))
    assert run["status"] == "aborted" and run["n_calib_seg"] == 0 and run["n_running_seg"] == 0


def test_abort_during_first_rest(data_root, tmp_path):
    runner = eng.FeedbackRunner(subj="001", root=data_root, out_root=tmp_path, run_id="r")
    gate = GateClock(runner.abort, lo=243.5)  # block0 后 30s rest 窗内
    # 先让引擎进 rest 再置 abort
    def setter():
        while runner.bus.snapshot is None or runner.bus.snapshot.get("phase") != "rest":
            if runner.outcome is not None:
                return
            time.sleep(0.001)
        runner.abort.set()
    threading.Thread(target=setter, daemon=True).start()
    runner.start(sleep_fn=gate.sleep, clock=gate.now)
    runner.join(10.0)
    assert not runner.alive
    assert runner.outcome["reason"] == "user_abort"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "r")
    assert sum(r["type"] == "seg" for r in rows) == 600
    assert rows[-1]["blocks_completed"] == 1
    assert any(r["type"] == "phase" and r["phase"] == "rest_start" for r in rows)
    assert not any(r["type"] == "phase" and r["phase"] == "rest_end" for r in rows)


def test_incomplete_data_mid_block(data_root, tmp_path, monkeypatch):
    # 截断到 1500 段（300s）：block0 完整、block1 中途耗尽
    orig = io_bids.load_channels_uV
    def truncated(subj, names, root=None):
        d = orig(subj, names, root)
        return {c: a[:1500 * 50] for c, a in d.items()}
    monkeypatch.setattr(eng.io_bids, "load_channels_uV", truncated)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="i", sleep_fn=clk.sleep, clock=clk.now)
    assert out["reason"] == "incomplete_data"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "i")
    assert rows[-1]["blocks_completed"] == 1 and rows[-1]["k"] == 1499
    run = json.loads((tmp_path / "feedback" / "sub-001" / "i" / "run.json").read_text("utf-8"))
    assert run["status"] == "aborted" and run["reason"] == "incomplete_data"
    assert run["n_running_seg"] == sum(r["type"] == "seg" for r in rows)


def test_freeze_failed_writes_aborted_no_calib_rows(data_root, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise tm.FreezeFailed("x")
    monkeypatch.setattr(tm, "calibrate_theta", boom)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="ff", sleep_fn=clk.sleep, clock=clk.now)
    assert out["reason"] == "freeze_failed"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "ff")
    assert rows[-1]["type"] == "aborted" and rows[-1]["reason"] == "freeze_failed"
    assert not any(r["type"] == "calib_seg" for r in rows)


def test_engine_error_unexpected_exception(data_root, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(qc, "p2p_uv", boom)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path,
                           run_id="ee", sleep_fn=clk.sleep, clock=clk.now)
    assert out["status"] == "aborted" and out["reason"] == "engine_error"
    rows = _rows(tmp_path / "feedback" / "sub-001" / "ee")
    assert rows[-1]["error"] == "RuntimeError"
    run = json.loads((tmp_path / "feedback" / "sub-001" / "ee" / "run.json").read_text("utf-8"))
    assert run["reason"] == "engine_error"
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err


def test_runner_baseexception_lands_as_engine_error(data_root, tmp_path, monkeypatch):
    """FeedbackRunner._target 必须捕获 BaseException（含 KeyboardInterrupt），
    落 engine_error 终态，不能让线程悄悄死掉导致 watcher 误判正常结束。"""
    def boom(*a, **k):
        raise KeyboardInterrupt("simulated SIGINT inside engine")
    monkeypatch.setattr(eng, "run_feedback", boom)
    runner = eng.FeedbackRunner(subj="001", root=data_root, out_root=tmp_path, run_id="be")
    runner.start()
    runner.join(5.0)
    assert not runner.alive, "引擎线程未在 5s 内结束"
    assert runner.outcome is not None, "outcome 为 None —— 异常没被 _target 捕获"
    assert runner.outcome["status"] == "aborted"
    assert runner.outcome["reason"] == "engine_error"
    assert runner.outcome["run_dir"] == runner.run_dir


def test_preflight_failure_writes_nothing(data_root, tmp_path):
    with pytest.raises(FileNotFoundError):
        eng.run_feedback(subj="999", root=data_root, out_root=tmp_path, run_id="z")
    assert not (tmp_path / "feedback").exists()


def test_preloaded_chans_bypasses_preflight_and_runs_ok(data_root, tmp_path, monkeypatch):
    """CLI 接缝：chans 预载后引擎不再碰 preflight/load（CLI 在 uvicorn 前完成它们）。"""
    raw = io_bids.load_channels_uV("001", list(eng.ALL_CH), data_root)
    short = {c: a[:4200 * 50] for c, a in raw.items()}  # 恰好 4200 段=600+3600

    def boom(*a, **k):
        raise AssertionError("预载 chans 时不应再调 preflight")

    monkeypatch.setattr(eng.io_bids, "preflight", boom)
    clk = FakeClock()
    out = eng.run_feedback(subj="001", root=data_root, out_root=tmp_path, run_id="p",
                           sleep_fn=clk.sleep, clock=clk.now, chans=short)
    assert out["status"] == "ok"
    assert len(_rows(tmp_path / "feedback" / "sub-001" / "p")) == 4225


def test_finite_or_none_guard():
    assert eng._finite_or_none(float("nan")) is None
    assert eng._finite_or_none(float("inf")) is None
    assert eng._finite_or_none(None) is None
    assert eng._finite_or_none(1.5) == 1.5


def test_calibration_buffer_footprint_under_10mb():
    # spec §4.1：三路 30000 float64 校准缓冲 = 720,000B ≈ 0.72MB，上限 < 10MB
    n_bytes = 3 * int(config.QC_CALIBRATION_SEC * config.SFREQ) * 8
    assert n_bytes == 720_000
    assert n_bytes < 10_000_000


def test_independent_recompute_matches_jsonl(data_root, tmp_path):
    """spec §7.3：只读 jsonl+meta 参数+原始数据，用公开原语重算阈值与各块结果。"""
    _, rd, _ = _run(data_root, tmp_path)
    rows = _rows(rd)
    meta = rows[0]
    freeze = next(r for r in rows if r["type"] == "freeze")
    names = [meta["target"]["channel"], *config.EOG_PROXY_CHANNELS]
    raw = io_bids.load_channels_uV("001", names, data_root)
    ff = {c: causal_filter.bandpass_oneshot(raw[c]) for c in names}
    n_cal = int(config.QC_CALIBRATION_SEC * config.SFREQ)
    b1, b2 = freeze["b_fp1"], freeze["b_fp2"]
    cal_clean = ff["Fz"][:n_cal] - b1 * ff["FP1"][:n_cal] - b2 * ff["FP2"][:n_cal]
    cal_valid = []
    for j in range(600):
        sl = slice(j * 50, j * 50 + 50)
        cal_valid.append(qc.segment_verdict(
            ff["Fz"][:n_cal][sl], ff["FP1"][:n_cal][sl], ff["FP2"][:n_cal][sl],
            b1, b2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV)["valid"])
    fr = tm.calibrate_theta(cal_clean, np.asarray(cal_valid, dtype=bool))
    assert fr["theta_thr_uV2"] == pytest.approx(freeze["theta_thr_uV2"], abs=1e-10)

    stream = tm.ThetaStream(zf=fr["zf"])
    smoother = tm.ValidOnlySmoother()
    decider = tm.HitDecider(fr["theta_thr_uV2"], config.TARGET.direction)
    got_rates = []
    seg_rows = [r for r in rows if r["type"] == "seg"]
    for b in range(6):
        nv = nh = 0
        for j in range(600):
            k = b * 600 + j
            s0, s1 = n_cal + k * 50, n_cal + (k + 1) * 50
            clean = ff["Fz"][s0:s1] - b1 * ff["FP1"][s0:s1] - b2 * ff["FP2"][s0:s1]
            _, p = stream.push_segment(clean)
            v = qc.segment_verdict(ff["Fz"][s0:s1], ff["FP1"][s0:s1], ff["FP2"][s0:s1],
                                   b1, b2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV)
            sm = smoother.update(p, v["valid"])
            hit = None if not v["valid"] or sm is None else decider.hit(sm)
            row = seg_rows[k]
            assert row["theta_pow_uV2"] == pytest.approx(p, rel=1e-12)
            assert row["smoothed_uV2"] == (None if sm is None else pytest.approx(sm, rel=1e-12))
            assert row["hit"] == hit
            if v["valid"]:
                nv += 1
                if hit:
                    nh += 1
        got_rates.append(round(nh / nv, 6))
    assert got_rates == [r["hit_rate"] for r in rows if r["type"] == "block_summary"]
    assert got_rates == rows[-1]["block_rates"]
