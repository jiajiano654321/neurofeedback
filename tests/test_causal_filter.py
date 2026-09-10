"""流式因果带通：分块==整段单向 lfilter；追加未来数据不改过去（C3）。"""
import pathlib

import numpy as np
import pytest

from bci_sys.replay import causal_filter as cf


def test_chunked_streaming_equals_oneshot():
    rng = np.random.default_rng(0)
    # 含大直流偏移（实测 ~1.44e6 µV），验证启动后直流被高通干掉
    x = 1.44e6 + 20.0 * rng.standard_normal(250 * 5)

    stream = cf.CausalBandpass(["Fz"])
    ys = [
        stream.push_block({"Fz": x[s : s + 50]})["Fz"]
        for s in range(0, x.size, 50)
    ]
    y_chunk = np.concatenate(ys)

    np.testing.assert_allclose(y_chunk, cf.bandpass_oneshot(x), atol=1e-10)


def test_appending_future_data_does_not_change_past():
    """因果性（守 C3）：先喂 10 段记录输出；喂 20 0.4 秒的未来数据重跑，前 10 段逐点不变。"""
    rng = np.random.default_rng(3)
    x_short = rng.standard_normal(50 * 10) * 20.0
    x_long = np.concatenate([x_short, rng.standard_normal(50 * 10) * 20.0])

    f1 = cf.CausalBandpass(["c"])
    out_short = np.concatenate([
        f1.push_block({"c": x_short[s : s + 50]})["c"] for s in range(0, 500, 50)
    ])
    f2 = cf.CausalBandpass(["c"])
    out_long = np.concatenate([
        f2.push_block({"c": x_long[s : s + 50]})["c"] for s in range(0, 1000, 50)
    ])
    np.testing.assert_allclose(out_short, out_long[:500], atol=1e-12)


def test_multichannel_states_are_independent():
    rng = np.random.default_rng(4)
    xs = {c: rng.standard_normal(50 * 4) * (10 + i * 5)
          for i, c in enumerate(["Fz", "FP1", "FP2"])}
    stream = cf.CausalBandpass(["Fz", "FP1", "FP2"])
    out = {c: [] for c in xs}
    for s in range(0, 200, 50):
        block = stream.push_block({c: xs[c][s : s + 50] for c in xs})
        for c in xs:
            out[c].append(block[c])
    for c in xs:
        np.testing.assert_allclose(
            np.concatenate(out[c]), cf.bandpass_oneshot(xs[c]), atol=1e-10
        )


def test_dc_offset_and_startup_transient():
    # 常数 1.44e6（实测直流偏移量级）阶跃输入：
    # 零初始 zi 的 1Hz 高通启动瞬态实测 2s 时仍有 ~6200µV、~9s 才衰减完，
    # 这正是 QC 剔除前 10s warmup 的原因（数值已实测，勿照手册反推）。
    x = np.full(250 * 12, 1.44e6)
    y = cf.bandpass_oneshot(x)
    assert np.abs(y[: 250 * 10]).max() > 100.0   # warmup 期内确有大瞬态
    assert np.abs(y[250 * 10 :]).max() < 1e-2    # 10s 后残差≈0（浮点舍入底噪）


def test_module_never_calls_filtfilt():
    # 只禁止【调用】零相位滤波；docstring 里的警示文字不算违规
    import ast

    tree = ast.parse(pathlib.Path(cf.__file__).read_text(encoding="utf-8"))
    attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert "filtfilt" not in attrs


def test_empty_block_does_not_corrupt_state():
    # scipy lfilter 对空输入会返回错误非零 zf；空块必须不触碰 zi，
    # 之后输出仍与「无空块的整段」逐点一致。
    rng = np.random.default_rng(8)
    x = rng.standard_normal(50 * 5) * 20.0  # 5 块数据，中间插一次空调用
    f = cf.CausalBandpass(["c"])
    outs = []
    for k, s in enumerate(range(0, x.size, 50)):
        if k == 2:
            # 额外插一次空块（不替换数据），验证 zi 不被污染
            f.push_block({"c": np.empty(0)})
        outs.append(f.push_block({"c": x[s:s + 50]})["c"])
    got = np.concatenate(outs)
    np.testing.assert_allclose(got, cf.bandpass_oneshot(x), atol=1e-10)


def test_2d_or_mismatched_block_raises_before_state_change():
    rng = np.random.default_rng(9)
    f = cf.CausalBandpass(["Fz", "FP1"])
    good = rng.standard_normal(50)
    f.push_block({"Fz": good, "FP1": good})  # 先推进一个正常块建立状态
    hp_zi_before = f._hp["Fz"].zi.copy()
    with pytest.raises(ValueError):
        f.push_block({"Fz": good, "FP1": good.reshape(25, 2)})  # 等 size 但二维
    with pytest.raises(ValueError):
        f.push_block({"Fz": good, "FP1": good[:10]})            # 长度不等
    np.testing.assert_array_equal(f._hp["Fz"].zi, hp_zi_before)  # 状态未被污染
