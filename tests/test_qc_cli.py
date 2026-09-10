"""qc-monitor 命令行入口的子进程测试（退出码 0/2、产物、零垃圾）。"""
import json
import subprocess
import sys


def test_cli_runs_limited_qc(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.qc_cli",
         "--subj", "001", "--duration-sec", "2",
         "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "CAL" in r.stdout
    run_dir = tmp_path / "qc" / "sub-001"
    assert run_dir.exists()
    js = list(run_dir.rglob("qc.jsonl"))
    assert len(js) == 1
    rows = [json.loads(l) for l in js[0].read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "summary" and rows[-1]["n_running"] == 0
    assert list(run_dir.rglob("valid_rate_timeline.png"))  # ok 路径必出图


def test_cli_bad_subject_exits_2_and_writes_nothing(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.qc_cli",
         "--subj", "999", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 2
    assert "PREFLIGHT" in r.stderr or "preflight" in r.stderr.lower()
    assert not (tmp_path / "qc").exists()
