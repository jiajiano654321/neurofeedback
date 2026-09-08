import pandas as pd

from bci_sys.offline import probe


def test_gate_passes_strictly_monotonic():
    assert probe.gate_passes({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}) is True


def test_gate_fails_flat_and_nonmonotonic():
    assert probe.gate_passes({1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0}) is False  # 相等不算
    assert probe.gate_passes({1: 1.0, 2: 3.0, 3: 2.0, 4: 4.0}) is False


def test_gate_fails_when_load_missing():
    assert probe.gate_passes({1: 1.0, 2: 2.0, 3: 3.0}) is False  # 缺 4-back


def test_median_by_load():
    df = pd.DataFrame({"load": [1, 1, 2, 2], "theta_uV2": [1.0, 3.0, 10.0, 20.0]})
    med = probe.median_by_load(df)
    assert med[1] == 2.0 and med[2] == 15.0


def test_run_probe_sub001_passes_and_writes_outputs(data_root, tmp_path):
    ok = probe.run_probe("001", root=data_root, out_dir=tmp_path)
    assert ok is True  # 硬门控：1→4-back 单调上升
    d = tmp_path / "theta" / "001"
    csv = d / "theta_by_load.csv"
    png = d / "theta_by_load.png"
    assert csv.exists() and png.exists()
    out = pd.read_csv(csv)
    assert list(out["load"]) == [1, 2, 3, 4]
    assert list(out.columns) == ["load", "n_epochs", "median_theta_uV2"]
    assert (out["n_epochs"] > 90).all()


def test_run_all_writes_summary(data_root, tmp_path):
    results = probe.run_all(root=data_root, out_dir=tmp_path, subs=["001", "002"])
    assert len(results) == 2
    assert (tmp_path / "theta" / "all" / "summary.csv").exists()
    assert (tmp_path / "theta" / "all" / "summary.png").exists()
