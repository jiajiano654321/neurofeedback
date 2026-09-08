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
