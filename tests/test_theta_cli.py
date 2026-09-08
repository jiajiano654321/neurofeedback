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


def test_theta_cli_all_always_exit0(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--all", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0  # --all 纯参考，退出码恒 0
    assert (tmp_path / "theta" / "all" / "summary.csv").exists()


def test_theta_cli_requires_subj_or_all(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.offline.cli",
         "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode != 0
