"""IT4 冻结常量（spec §5.3，C2：运行期无修改入口）。"""

from bci_sys import config


def test_it4_frozen_constants():
    assert config.THETA_SMOOTH_SEGS == 5
    assert config.THETA_QUANTILE == 0.20
    assert config.THETA_TARGET_RATE == 0.80
    assert config.THETA_MIN_CALIB_SEGS == 50
    assert config.FEEDBACK_N_BLOCKS == 6
    assert config.FEEDBACK_BLOCK_SEC == 120.0
    assert config.FEEDBACK_REST_SEC == 30.0
    assert config.FEEDBACK_FREEZE_CARD_SEC == 3.0
    assert config.SESSION_WALL_CAP_SEC == 1500.0


def test_it4_reused_constants_unchanged():
    # 复用项不得在 IT4 被顺手改掉
    assert config.THETA_BAND_HZ == (4.0, 8.0)
    assert config.QC_CALIBRATION_SEC == 120.0
    assert config.QC_FILTER_WARMUP_SEC == 10.0
    assert config.QC_L1_P2P_UV == 100.0
    assert config.QC_L3_EOG_P2P_UV == 200.0
    assert config.TARGET.channel == "Fz"
    assert config.TARGET.direction == "up"
    assert config.SEGMENT_SAMPLES == 50 and config.SFREQ == 250.0


def test_it4_segment_counts():
    cal = int(round(config.QC_CALIBRATION_SEC / config.SEGMENT_SEC))
    per_block = int(round(config.FEEDBACK_BLOCK_SEC / config.SEGMENT_SEC))
    assert cal == 600
    assert per_block == 600
    assert config.FEEDBACK_N_BLOCKS * per_block == 3600
    assert config.QC_CALIBRATION_SEC + config.FEEDBACK_N_BLOCKS * config.FEEDBACK_BLOCK_SEC == 840.0
