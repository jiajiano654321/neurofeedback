import json
import subprocess
import sys


def test_cli_runs_limited_replay(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.cli",
         "--subj", "001", "--duration-sec", "2",
         "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "rate" in r.stdout
    run_dir = tmp_path / "replay" / "sub-001"
    assert run_dir.exists()
    js = list(run_dir.rglob("replay.jsonl"))
    assert len(js) == 1
    rows = [json.loads(l) for l in js[0].read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "summary" and rows[-1]["n_segments"] == 10


def test_cli_bad_subject_exits_nonzero_and_writes_nothing(data_root, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "bci_sys.replay.cli",
         "--subj", "999", "--data-root", str(data_root), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode != 0
    assert "PREFLIGHT" in r.stderr or "preflight" in r.stderr.lower()
    assert not (tmp_path / "replay").exists()
