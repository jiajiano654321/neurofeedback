import numpy as np
import pandas as pd

from bci_sys import config
from bci_sys.replay import segments


def test_segmenter_basic_shape_and_times():
    x = np.arange(150, dtype=float)  # 3 整段
    segs = list(segments.iter_segments(x, sfreq=250.0, seg_samples=50))
    assert len(segs) == 3
    assert segs[0].seg == 0
    assert np.allclose(segs[0].samples, np.arange(50))
    assert segs[0].t0 == 0.0
    assert abs(segs[0].t1 - 0.2) < 1e-9
    assert np.allclose(segs[2].samples, np.arange(100, 150))


def test_segmenter_tail_not_emitted():
    x = np.arange(130, dtype=float)  # 2 整段 + 30 点尾巴
    segs = list(segments.iter_segments(x, sfreq=250.0, seg_samples=50))
    assert len(segs) == 2
    # 尾巴 = 130/250 - 2*0.2 = 0.12 s
    assert abs(segments.tail_seconds(x.size, sfreq=250.0, seg_samples=50) - 0.12) < 1e-9


def test_segmenter_duration_limit_at_segment_boundary():
    x = np.arange(250, dtype=float)  # 5 段 = 1.0 s
    segs = list(segments.iter_segments(x, sfreq=250.0, seg_samples=50, duration_sec=0.4))
    assert len(segs) == 2  # 0.4 s = 2 段，段边界截断
    assert segs[-1].seg == 1


def test_segmenter_duration_zero_means_full():
    x = np.arange(250, dtype=float)
    assert len(list(segments.iter_segments(x, sfreq=250.0, seg_samples=50, duration_sec=0.0))) == 5


def test_segment_events_half_open_interval():
    ev = pd.DataFrame({"onset": [0.0, 0.2, 0.25, 0.4], "trial_type": ["a", "b", "c", "d"]})
    # 段 [0.2, 0.4)：含 0.2、0.25，不含 0.0、0.4（左闭右开，边界不重复计）
    got = segments.segment_events(ev, t0=0.2, t1=0.4)
    assert list(got["trial_type"]) == ["b", "c"]


def test_sub001_full_run_is_4792_segments(data_root):
    from bci_sys import io_bids

    fz = io_bids.load_fz_uV("001", data_root)
    segs = list(
        segments.iter_segments(fz, sfreq=config.SFREQ, seg_samples=config.SEGMENT_SAMPLES)
    )
    assert len(segs) == 4792
    assert all(s.samples.size == 50 for s in segs)
    assert abs(segments.tail_seconds(fz.size) - 0.112) < 0.005


# --- 终审修复 1：全量真实点数、限时 tail 语义、sfreq 贯穿、负 duration ---


def test_full_real_sample_counts_match_expected():
    """全量 sub-001 信号 239628 点 → 4792 段、tail≈0.112s。"""
    assert segments.n_planned_segments(239628, 50) == 4792
    assert abs(segments.tail_seconds(239628) - 0.112) < 0.005


def test_duration_at_segment_boundary_tail_is_zero():
    """限时 60s 落在段边界 → 300 段、tail=0.0。"""
    assert segments.n_planned_segments(239628, 50, 60.0, sfreq=250.0) == 300
    assert abs(segments.tail_seconds(239628, 250.0, 50, 60.0) - 0.0) < 1e-12


def test_duration_mid_segment_tail_is_residual():
    """duration 0.3s 落在段中间 → available=75 点、1 段、tail=25/250=0.1s。"""
    assert segments.n_planned_segments(239628, 50, 0.3, sfreq=250.0) == 1
    assert abs(segments.tail_seconds(239628, 250.0, 50, 0.3) - 0.1) < 1e-9


def test_negative_duration_raises_value_error():
    """负 duration 必须显式抛错，不能静默当全量。"""
    import pytest

    with pytest.raises(ValueError):
        segments.n_planned_segments(1000, 50, -1.0, sfreq=250.0)
    with pytest.raises(ValueError):
        segments.tail_seconds(1000, 250.0, 50, -1.0)


def test_sfreq_threads_through_duration_calculation():
    """不同 sfreq 下 duration 截断按该 sfreq 计算。

    500 Hz、seg_samples=100、duration_sec=1.0
    → available = int(1.0 * 500) = 500 点
    → n_planned = 500 // 100 = 5
    """
    assert segments.n_planned_segments(10000, 100, 1.0, sfreq=500.0) == 5
    # 限时 0.25s（125 点）→ 1 段、tail=25/500=0.05s
    assert segments.n_planned_segments(10000, 100, 0.25, sfreq=500.0) == 1
    assert abs(segments.tail_seconds(10000, 500.0, 100, 0.25) - 0.05) < 1e-9
