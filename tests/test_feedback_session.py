"""IT4 会话状态机 + qc.segment_verdict 复用（spec §4.3–4.5、§5）。"""

import numpy as np
import pytest

from bci_sys import config
from bci_sys.replay import causal_filter, feedback, qc, theta_metric as tm


def test_segment_verdict_matches_frozen_judge(data_root):
    """校准重建用的纯函数与冻结后 QcJudge 的逐段裁决必须同源同结果。"""
    chans = {}
    import bci_sys.io_bids as io_bids
    raw = io_bids.load_channels_uV("001", ["Fz", "FP1", "FP2"], data_root)
    n = int(config.QC_CALIBRATION_SEC * config.SFREQ) + 50 * 4
    bp = causal_filter.CausalBandpass(("Fz", "FP1", "FP2"))
    filtered = bp.push_block({c: raw[c][:n] for c in ("Fz", "FP1", "FP2")})
    judge = qc.QcJudge()
    for k in range(n // 50):
        sl = slice(k * 50, (k + 1) * 50)
        v = judge.push((k + 1) * 0.2, {c: filtered[c][sl] for c in ("Fz", "FP1", "FP2")})
    assert judge.frozen
    for k in range(600, n // 50):
        sl = slice(k * 50, (k + 1) * 50)
        got = qc.segment_verdict(
            filtered["Fz"][sl], filtered["FP1"][sl], filtered["FP2"][sl],
            judge.b_fp1, judge.b_fp2, config.QC_L1_P2P_UV, config.QC_L3_EOG_P2P_UV,
        )
        exp = judge.push((k + 1) * 0.2, {c: filtered[c][sl] for c in ("Fz", "FP1", "FP2")})
        assert got["valid"] == exp["valid"]
        assert got["reasons"] == exp["reasons"]
        assert got["fz_clean_p2p"] == pytest.approx(exp["fz_clean_p2p"])
        assert got["eog_p2p"] == pytest.approx(exp["eog_p2p"])


def _freeze_info(n=550, thr=16.7, rate=0.8, b1=0.248, b2=0.378):
    return {"n_thr_segs": n, "theta_thr_uV2": thr, "calib_hit_rate": rate,
            "b_fp1": b1, "b_fp2": b2}


def test_freeze_note_and_block_lifecycle():
    s = feedback.FeedbackSession()
    assert s.phase == "calibration"
    s.note_freeze(_freeze_info())
    assert s.phase == "freeze_card" and s.freeze_info["theta_thr_uV2"] == 16.7

    row = s.begin_block(120.0)
    assert row == {"type": "phase", "phase": "block_start", "block_idx": 0, "t1": 120.0}
    for j in range(600):
        s.record_segment(valid=(j % 4 != 0), power_finite=True, hit=(j % 2 == 0))
    # valid=450（j%4∈{1,2,3}）；命中=有效且偶数=j%4==2 → 150
    summ = s.end_block_if_due(240.0)
    assert summ["type"] == "block_summary" and summ["block_idx"] == 0
    assert summ["n_valid"] == 450 and summ["n_invalid"] == 150
    assert summ["n_hit"] == 150
    assert summ["hit_rate"] == round(150 / 450, 6)
    assert s.blocks_completed == 1 and s.block_rates[-1] == summ["hit_rate"]


def test_rest_does_not_advance_any_state():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    for _ in range(600):
        s.record_segment(valid=True, power_finite=True, hit=True)
    s.end_block_if_due(240.0)
    r1 = s.begin_rest(30.0)
    assert r1 == {"type": "phase", "phase": "rest_start", "rest_idx": 0, "rest_sec": 30.0}
    counts = (s.total.n_valid, s.total.n_hit, s.blocks_completed)
    r2 = s.end_rest(30.0)
    assert r2 == {"type": "phase", "phase": "rest_end", "rest_idx": 0, "rest_sec": 30.0}
    assert (s.total.n_valid, s.total.n_hit, s.blocks_completed) == counts
    row = s.begin_block(240.0)
    assert row["block_idx"] == 1 and s.seg_in_block == 0


def test_counters_frozen_on_invalid_segment_c5():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    s.record_segment(valid=True, power_finite=True, hit=True)
    s.record_segment(valid=False, power_finite=True, hit=None)
    s.record_segment(valid=True, power_finite=False, hit=None)  # 实测不可达守卫：不动计数
    assert s.total.n_valid == 1 and s.total.n_hit == 1 and s.total.n_invalid == 1
    assert s.seg_in_block == 3


def test_block_rate_zero_division_returns_zero():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    for _ in range(600):
        s.record_segment(valid=False, power_finite=True, hit=None)
    summ = s.end_block_if_due(240.0)
    assert summ["n_valid"] == 0 and summ["hit_rate"] == 0.0


def test_full_six_blocks_final_totals():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    for b in range(6):
        s.begin_block(120.0 + b * 120.0)
        for j in range(600):
            s.record_segment(valid=(j != 7), power_finite=True, hit=(j % 3 == 0 and j != 7))
        s.end_block_if_due(240.0 + b * 120.0)
        if b < 5:
            s.begin_rest(30.0)
            s.end_rest(30.0)
    final = s.finish()
    assert final["type"] == "final" and final["status"] == "ok"
    assert final["n_valid"] == 6 * 599 and final["n_invalid"] == 6
    assert final["n_hit"] == sum(1 for j in range(600) if j % 3 == 0 and j != 7) * 6
    assert len(final["block_rates"]) == 6
    assert final["overall_hit_rate"] == round(final["n_hit"] / final["n_valid"], 6)
    assert s.phase == "finished"


def test_freeze_with_too_few_segments_raises():
    s = feedback.FeedbackSession()
    with pytest.raises(tm.FreezeFailed):
        s.note_freeze(_freeze_info(n=49))
    assert s.phase == "calibration"  # 未进入冻结卡


def test_terminate_rows_and_reason_whitelist():
    s = feedback.FeedbackSession()
    s.note_freeze(_freeze_info())
    s.begin_block(120.0)
    s.record_segment(valid=True, power_finite=True, hit=False)
    row = s.terminate("user_abort", k=600, t1=120.2)
    assert row["type"] == "aborted" and row["status"] == "aborted"
    assert row["reason"] == "user_abort" and row["k"] == 600
    assert row["blocks_completed"] == 0 and s.phase == "aborted"
    with pytest.raises(ValueError):
        feedback.FeedbackSession().terminate("nope", k=0, t1=0.0)


def test_wall_cap_terminate():
    s = feedback.FeedbackSession(wall_cap_sec=100.0)
    assert s.wall_cap_terminate(99.9, k=10, t1=2.2) is None
    row = s.wall_cap_terminate(100.1, k=10, t1=2.2)
    assert row["reason"] == "wall_cap"
    # 已终态后不再出第二行
    assert s.wall_cap_terminate(999.0, k=11, t1=2.4) is None


def test_engine_error_row_carries_error_name():
    s = feedback.FeedbackSession()
    row = s.terminate("engine_error", k=3, t1=0.8, error="RuntimeError")
    assert row["error"] == "RuntimeError"
