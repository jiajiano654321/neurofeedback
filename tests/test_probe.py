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


def test_run_all_missing_data_per_subject_tolerance(tmp_path):
    """--all 逐被试容错：缺数据不抛异常、status=error、仍写出 summary.csv。"""
    rows = probe.run_all(
        root=tmp_path / "nonexistent", out_dir=tmp_path / "out", subs=["001", "002"]
    )
    assert len(rows) == 2
    for r in rows:
        assert r["status"] == "error"
        assert r["pass"] is False
        assert r["m1"] is None and r["m2"] is None and r["m3"] is None and r["m4"] is None
        assert r["error"] != ""
    # 即使全失败也要写文件
    csv_path = tmp_path / "out" / "theta" / "all" / "summary.csv"
    assert csv_path.exists()
    import pandas as pd

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["subj", "status", "m1", "m2", "m3", "m4", "pass", "error"]
    assert list(df["status"]) == ["error", "error"]


def test_run_all_recommended_subs_mixed_status(data_root, tmp_path):
    """真实数据：推荐被试中 sub-001 ok/pass，sub-014 error（单位超门），共 7 行。"""
    from bci_sys import config

    rows = probe.run_all(root=data_root, out_dir=tmp_path, subs=config.RECOMMENDED_SUBS)
    assert len(rows) == 7
    by_sub = {r["subj"]: r for r in rows}
    # sub-001：正常通过
    assert by_sub["001"]["status"] == "ok"
    assert by_sub["001"]["pass"] is True
    assert by_sub["001"]["error"] == ""
    # sub-014：单位超门，被 preflight 拦下
    assert by_sub["014"]["status"] == "error"
    assert by_sub["014"]["pass"] is False
    assert by_sub["014"]["m1"] is None
    assert "std" in by_sub["014"]["error"] or "EEG" in by_sub["014"]["error"]
    # 产物存在
    assert (tmp_path / "theta" / "all" / "summary.csv").exists()
    assert (tmp_path / "theta" / "all" / "summary.png").exists()
