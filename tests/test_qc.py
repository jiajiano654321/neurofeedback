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
