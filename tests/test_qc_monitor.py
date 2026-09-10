import json
from pathlib import Path

from bci_sys.replay import pace, qc_monitor


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class AbortClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if self.t >= 0.6:  # 第 3 段结束时中止
            raise KeyboardInterrupt


def _read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def _run_dir(tmp_path, rid="fixed"):
    return tmp_path / "qc" / "sub-001" / rid


def test_qc_short_run_meta_seg_summary_schema(data_root, tmp_path):
    clk = FakeClock()
    status = qc_monitor.run_qc(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "ok"
    rows = _read_jsonl(_run_dir(tmp_path) / "qc.jsonl")
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "meta" and kinds[-1] == "summary"
    assert kinds.count("seg") == 10
    assert "calib" not in kinds  # 2s 短回放不到校准边界

    meta = rows[0]
    assert meta["schema_version"] == "1.0"
    assert meta["subj"] == "001" and meta["sfreq"] == 250.0
    assert meta["segment_sec"] == 0.2 and meta["n_segments_planned"] == 10
    assert meta["channels"] == {"Fz": "Fz", "FP1": "FP1", "FP2": "FP2"}
    assert meta["filter_band_hz"] == [1.0, 30.0]
    assert meta["calibration_sec"] == 120.0 and meta["filter_warmup_sec"] == 10.0
    assert meta["l1_p2p_uv"] == 100.0 and meta["l3_eog_p2p_uv"] == 200.0
    assert "code_version" in meta and meta["code_version"] != "unknown"

    for r in rows[1:-1]:
        assert r["phase"] == "calibration"
        assert r["valid"] is None and r["reasons"] == []
        assert r["fz_clean_p2p"] is None and r["eog_p2p"] is None

    summ = rows[-1]
    assert summ["n_running"] == 0 and summ["n_valid"] == 0 and summ["n_invalid"] == 0
    assert summ["n_l1"] == 0 and summ["n_l3"] == 0 and summ["n_l1_and_l3"] == 0
    assert summ["valid_rate"] is None  # 运行期 0 段 → null，不许 ZeroDivision

    run = json.loads((_run_dir(tmp_path) / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "ok"
    assert {"wall_started_iso", "wall_elapsed_s", "rate_ratio", "n_segments"} <= run.keys()
    assert (_run_dir(tmp_path) / "valid_rate_timeline.png").exists()


def test_qc_jsonl_byte_deterministic_and_wall_free(data_root, tmp_path):
    outs = []
    for rid in ("a", "b"):
        clk = FakeClock()
        qc_monitor.run_qc(
            "001", duration_sec=2.0, root=data_root, out_root=tmp_path / rid,
            run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
        )
        p = tmp_path / rid / "qc" / "sub-001" / "fixed" / "qc.jsonl"
        outs.append(p.read_bytes())
        text = p.read_text(encoding="utf-8")
        assert "wall" not in text and "rate_ratio" not in text  # A4：墙钟只进 run.json
    assert outs[0] == outs[1]


def test_qc_abort_writes_run_json_no_summary(data_root, tmp_path):
    clk = AbortClock()
    status = qc_monitor.run_qc(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="x", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "aborted"
    rd = _run_dir(tmp_path, "x")
    rows = _read_jsonl(rd / "qc.jsonl")
    assert rows[-1]["type"] != "summary"
    run = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "aborted"
    assert not (rd / "valid_rate_timeline.png").exists()  # 仅 ok 才出图


def test_qc_preflight_failure_writes_nothing(data_root, tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        qc_monitor.run_qc("999", root=data_root, out_root=tmp_path, run_id="z")
    assert not (tmp_path / "qc").exists()


def test_qc_real_data_130s_freeze_and_counters(data_root, tmp_path):
    """真数据结构验收（不设有效率断言）：冻结 b 有限且 0.1–0.7；计数自洽。"""
    clk = FakeClock()  # 假时钟：130s 数据瞬时跑完
    status = qc_monitor.run_qc(
        "001", duration_sec=130.0, root=data_root, out_root=tmp_path,
        run_id="real", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "ok"
    rd = _run_dir(tmp_path, "real")
    rows = _read_jsonl(rd / "qc.jsonl")

    calibs = [r for r in rows if r["type"] == "calib"]
    assert len(calibs) == 1
    c = calibs[0]
    assert c["t_sec"] == 120.0 and c["n_fit_samples"] == 27500
    for key in ("b_fp1", "b_fp2"):
        assert c[key] == c[key]  # 非 NaN
        assert 0.1 <= c[key] <= 0.7  # 实测合理区间（结构验收，非疗效结论）

    segs = [r for r in rows if r["type"] == "seg"]
    assert len(segs) == 650
    assert sum(r["phase"] == "calibration" for r in segs) == 600
    running = [r for r in segs if r["phase"] == "running"]
    assert len(running) == 50
    assert all(r["valid"] in (True, False) for r in running)
    assert all(set(r["reasons"]) <= {"L1", "L3"} for r in running)

    summ = rows[-1]
    assert summ["type"] == "summary"
    assert summ["n_running"] == 50
    assert summ["n_valid"] + summ["n_invalid"] == 50
    assert summ["n_l1"] == sum("L1" in r["reasons"] for r in running)
    assert summ["n_l3"] == sum("L3" in r["reasons"] for r in running)
    assert summ["n_l1_and_l3"] == sum(
        {"L1", "L3"} <= set(r["reasons"]) for r in running
    )
    assert 0.0 <= summ["valid_rate"] <= 1.0
    assert (rd / "valid_rate_timeline.png").exists()
