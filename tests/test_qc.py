"""QC 纯逻辑：OLS 系数回收、warmup 剔除、非有限值硬错。"""
import numpy as np
import pytest

from bci_sys.replay import qc


def test_p2p_basic():
    assert qc.p2p_uv(np.array([3.0, -7.0, 10.0, 2.0])) == 17.0


def test_ols_recovers_known_coefficients():
    # 合成模型 Fz = 0.30*FP1 + 0.50*FP2 + 独立残差；120s @250Hz
    rng = np.random.default_rng(0)
    n = 250 * 120
    fp1 = rng.standard_normal(n) * 30.0
    fp2 = rng.standard_normal(n) * 25.0
    fz = 0.30 * fp1 + 0.50 * fp2 + rng.standard_normal(n) * 10.0
    b1, b2 = qc.fit_eog_coeffs(fz, fp1, fp2, start_sample=250 * 10)
    assert abs(b1 - 0.30) < 0.02
    assert abs(b2 - 0.50) < 0.02


def test_ols_warmup_exclusion_is_necessary_and_sufficient():
    """前 10s（2500 点）放【相关的】启动瞬态：含进去估错，剔除后估对。

    注意常数污染会被去均值消掉、不影响斜率，所以启动瞬态必须是
    Fz 与 FP1 相关、但比例错误（-3.0）的共同阶跃，才考得出 warmup 的作用。
    """
    rng = np.random.default_rng(1)
    n = 250 * 120
    fp1 = np.zeros(n)
    fp1[2500:] = rng.standard_normal(n - 2500) * 30.0
    fz = 0.40 * fp1 + rng.standard_normal(n) * 5.0
    fp1[:2500] = 5000.0     # 启动期共同大阶跃
    fz[:2500] = -15000.0    # Fz/FP1 比例 -3.0（错误系数，主导 OLS）
    b_bad, _ = qc.fit_eog_coeffs(fz, fp1, fp1 * 0, start_sample=0)
    b_good, _ = qc.fit_eog_coeffs(fz, fp1, fp1 * 0, start_sample=2500)
    assert abs(b_bad - 0.40) > 0.1
    assert abs(b_good - 0.40) < 0.02


def test_ols_rejects_nonfinite_and_bad_arguments():
    n = 1000
    fz = np.full(n, np.nan)
    with pytest.raises(ValueError):
        qc.fit_eog_coeffs(fz, np.zeros(n), np.zeros(n), start_sample=0)
    with pytest.raises(ValueError):
        qc.fit_eog_coeffs(np.zeros(n), np.zeros(n - 1), np.zeros(n), 0)
    with pytest.raises(ValueError):
        qc.fit_eog_coeffs(np.zeros(n), np.zeros(n), np.zeros(n), n + 1)


def test_ols_rejects_zero_rank_eog_and_accepts_one_flat_channel():
    rng = np.random.default_rng(10)
    n = 5000
    good = rng.standard_normal(n) * 30.0
    fz = 0.4 * good + rng.standard_normal(n) * 5.0
    flat = np.zeros(n)
    # 两路 EOG 全程恒定 → 设计矩阵零秩 → 硬拦（不许静默冻结成 b=0 不去眼电）
    with pytest.raises(ValueError, match="零秩"):
        qc.fit_eog_coeffs(fz, flat, flat, start_sample=0)
    # 一路恒定、另一路有效（秩1）仍可接受：好通道系数估出、平通道系数≈0
    b1, b2 = qc.fit_eog_coeffs(fz, good, flat, start_sample=0)
    assert abs(b1 - 0.4) < 0.02 and abs(b2) < 1e-8


def test_ols_inf_input_raises_without_runtime_warning():
    import warnings

    n = 100
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # 去均值不得先报警
        with pytest.raises(ValueError):
            qc.fit_eog_coeffs(
                np.full(n, np.inf), np.ones(n), np.full(n, 2.0), 0
            )


# ---------- QcJudge 状态机 ----------

def _blocks(fz, fp1, fp2, seg=50):
    for s in range(0, fz.size, seg):
        yield {"Fz": fz[s:s + seg], "FP1": fp1[s:s + seg], "FP2": fp2[s:s + seg]}


def _synthetic_120s(b1=0.25, b2=0.38, seed=5):
    rng = np.random.default_rng(seed)
    n = 250 * 120
    fp1 = rng.standard_normal(n) * 30.0
    fp2 = rng.standard_normal(n) * 25.0
    fz = b1 * fp1 + b2 * fp2 + rng.standard_normal(n) * 10.0
    return fz, fp1, fp2


def test_judge_calibration_then_freeze_then_running():
    fz, fp1, fp2 = _synthetic_120s()
    judge = qc.QcJudge()
    blocks = list(_blocks(fz, fp1, fp2))
    assert len(blocks) == 600

    calib_seen = 0
    for k, blk in enumerate(blocks):
        t1 = (k + 1) * 0.2
        v = judge.push(t1, blk)
        assert v["phase"] == "calibration"
        assert v["valid"] is None and v["reasons"] == []
        assert v["fz_clean_p2p"] is None and v["eog_p2p"] is None
        if v.get("calib") is not None:
            calib_seen += 1
            c = v["calib"]
            assert c["t_sec"] == 120.0
            assert c["fit_window_sec"] == [10.0, 120.0]
            assert c["n_fit_samples"] == 27500
            assert abs(c["b_fp1"] - 0.25) < 0.05
            assert abs(c["b_fp2"] - 0.38) < 0.05
    assert calib_seen == 1  # 恰在边界冻结一次
    assert judge.frozen

    # 第 601 段起进入运行期
    v = judge.push(120.2, {"Fz": np.zeros(50), "FP1": np.zeros(50), "FP2": np.zeros(50)})
    assert v["phase"] == "running" and v["valid"] is True and v["reasons"] == []
    assert v["fz_clean_p2p"] == 0.0 and v["eog_p2p"] == 0.0


def test_judge_running_verdicts_l1_l3_both():
    fz, fp1, fp2 = _synthetic_120s()
    judge = qc.QcJudge()
    for k, blk in enumerate(_blocks(fz, fp1, fp2)):
        judge.push((k + 1) * 0.2, blk)

    rng = np.random.default_rng(6)

    # L1：残差 p2p=101，EOG 两路干净
    fz_bad = np.linspace(0.0, 101.0, 50)
    v = judge.push(120.2, {"Fz": fz_bad, "FP1": np.zeros(50), "FP2": np.zeros(50)})
    assert v["valid"] is False and v["reasons"] == ["L1"]
    assert v["fz_clean_p2p"] == pytest.approx(101.0)
    assert v["eog_p2p"] == 0.0

    # L3：FP2 p2p=201，Fz 干净
    fp2_bad = np.linspace(0.0, 201.0, 50)
    v = judge.push(120.4, {"Fz": np.zeros(50), "FP1": np.zeros(50), "FP2": fp2_bad})
    assert v["valid"] is False and v["reasons"] == ["L3"]
    assert v["eog_p2p"] == pytest.approx(201.0)

    # L1 与 L3 同时命中 → 原因合并全记录（C5）
    fp1_bad = rng.standard_normal(50) * 200.0
    fp1_bad -= fp1_bad.min()  # p2p 远超 200
    v = judge.push(120.6, {"Fz": fz_bad, "FP1": fp1_bad, "FP2": np.zeros(50)})
    assert v["valid"] is False and set(v["reasons"]) == {"L1", "L3"}

    # 阈值边界：恰好 100 / 200 不算超标（严格 >）
    v = judge.push(120.8, {
        "Fz": np.linspace(0.0, 100.0, 50),
        "FP1": np.linspace(0.0, 200.0, 50), "FP2": np.zeros(50),
    })
    assert v["valid"] is True


def test_judge_nan_segment_is_never_marked_valid():
    # 非有限段不得被 nan>thr=False 静默判成有效（L1/L3 同时命中）
    fz, fp1, fp2 = _synthetic_120s()
    judge = qc.QcJudge()
    for k, blk in enumerate(_blocks(fz, fp1, fp2)):
        judge.push((k + 1) * 0.2, blk)
    nan_block = {c: np.full(50, np.nan) for c in ("Fz", "FP1", "FP2")}
    v = judge.push(120.2, nan_block)
    assert v["valid"] is False
    assert set(v["reasons"]) == {"L1", "L3"}

    # 仅 Fz 含 NaN、EOG 干净：只命中 L1，不冤枉 L3
    v = judge.push(120.4, {
        "Fz": np.full(50, np.nan), "FP1": np.zeros(50), "FP2": np.zeros(50),
    })
    assert v["valid"] is False and v["reasons"] == ["L1"]


def test_judge_short_run_never_freezes():
    # 数据短于 120s：全程 calibration，无 calib，无异常
    fz, fp1, fp2 = _synthetic_120s()
    judge = qc.QcJudge()
    for k in range(10):
        s = k * 50
        v = judge.push((k + 1) * 0.2,
                       {"Fz": fz[s:s+50], "FP1": fp1[s:s+50], "FP2": fp2[s:s+50]})
        assert v["phase"] == "calibration" and v.get("calib") is None
    assert not judge.frozen


def test_judge_freeze_raises_on_nonfinite_b():
    # 用小校准窗加速：0.4s = 100 点，warmup=0；输入 NaN → 冻结时 ValueError
    judge = qc.QcJudge(calibration_sec=0.4, warmup_sec=0.0)
    nan_block = {"Fz": np.full(50, np.nan), "FP1": np.zeros(50), "FP2": np.zeros(50)}
    judge.push(0.2, nan_block)
    with pytest.raises(ValueError, match="非有限"):
        judge.push(0.4, nan_block)


def test_judge_warmup_excludes_startup_transient():
    """judge 级：短窗合成，启动样本污染时 warmup 剔除后仍回收正确系数。"""
    rng = np.random.default_rng(7)
    n = 250  # 1.0s 校准
    fp1 = np.zeros(n)
    fp1[125:] = rng.standard_normal(125) * 30.0
    fz = 0.40 * fp1 + rng.standard_normal(n) * 2.0
    fp1[:125] = 5000.0    # 启动期共同大阶跃（比例错误的相关瞬态）
    fz[:125] = -15000.0
    judge = qc.QcJudge(calibration_sec=1.0, warmup_sec=0.5)
    blocks = list(_blocks(fz, fp1, np.zeros(n)))
    v = None
    for k, blk in enumerate(blocks):
        v = judge.push((k + 1) * 0.2, blk)
    assert judge.frozen
    assert abs(v["calib"]["b_fp1"] - 0.40) < 0.05
    assert v["calib"]["n_fit_samples"] == 125
