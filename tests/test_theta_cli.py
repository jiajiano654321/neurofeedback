import subprocess
import sys


def test_theta_cli_sub001_pass_exit0(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--subj", "001", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "PASS" in r.stdout
    assert (tmp_path / "theta" / "001" / "theta_by_load.png").exists()


def test_theta_cli_all_missing_data_exit0_with_qc_fail(tmp_path):
    """--all 缺数据不退出 2：逐被试容错，退出码恒 0，stdout 含 QC FAIL。"""
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--all", "--data-root", str(tmp_path / "no_such_dir"), "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0
    assert "QC FAIL" in r.stdout
    assert (tmp_path / "out" / "theta" / "all" / "summary.csv").exists()


def test_theta_cli_all_real_data_exit0_and_writes_outputs(data_root, tmp_path):
    """真实数据：--all 退出码 0，产出 summary.csv 与 summary.png。"""
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--all", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "theta" / "all" / "summary.csv").exists()
    assert (tmp_path / "theta" / "all" / "summary.png").exists()
    # sub-001 通过、sub-014 QC 失败
    assert "sub-001: PASS" in r.stdout
    assert "sub-014: QC FAIL" in r.stdout


def test_theta_cli_requires_subj_or_all(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode != 0
