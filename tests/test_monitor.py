import json
from pathlib import Path

from bci_sys.replay import monitor, pace


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt  # 不走真睡眠


def _read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def test_replay_writes_meta_seg_event_summary(data_root, tmp_path):
    clk = FakeClock()
    status = monitor.run_replay(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "ok"
    run_dir = tmp_path / "replay" / "sub-001" / "fixed"
    rows = _read_jsonl(run_dir / "replay.jsonl")
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "meta"
    assert kinds[-1] == "summary"
    assert kinds.count("seg") == 10  # 2.0 s = 10 段
    meta = rows[0]
    assert meta["sfreq"] == 250.0 and meta["fz_channel"] == "Fz"
    assert meta["n_segments_planned"] == 10
    assert "code_version" in meta and meta["code_version"] != "unknown"
    seg = next(r for r in rows if r["type"] == "seg")
    assert seg["seg"] == 0 and len(seg["fz_uV"]) == 50
    evs = [r for r in rows if r["type"] == "event"]
    assert any(e["kind"] == "dropped" for e in evs)
    summ = rows[-1]
    assert summ["n_segments"] == 10
    assert "wall_elapsed_s" not in summ and "rate_ratio" not in summ
    assert summ["n_dropped"] >= 1


def test_replay_jsonl_byte_deterministic(data_root, tmp_path):
    """A4：同参数两次回放，replay.jsonl 字节级一致。"""
    outs = []
    for rid in ["a", "b"]:
        clk = FakeClock()
        monitor.run_replay(
            "001", duration_sec=2.0, root=data_root, out_root=tmp_path / rid,
            run_id="fixed", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
        )
        outs.append((tmp_path / rid / "replay" / "sub-001" / "fixed" / "replay.jsonl").read_bytes())
    assert outs[0] == outs[1]


def test_run_json_contains_wall_and_status(data_root, tmp_path):
    clk = FakeClock()
    monitor.run_replay(
        "001", duration_sec=1.0, root=data_root, out_root=tmp_path,
        run_id="r", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    run = json.loads((tmp_path / "replay" / "sub-001" / "r" / "run.json").read_text())
    assert run["status"] == "ok"
    assert "wall_elapsed_s" in run and "rate_ratio" in run and "wall_started_iso" in run


def test_abort_writes_run_json_and_no_summary(data_root, tmp_path):
    """Ctrl+C：run.json=aborted，已写内容保留，不写 summary 行（D3 精神）。"""

    class AbortClock:
        def __init__(self):
            self.t = 0.0

        def now(self):
            return self.t

        def sleep(self, dt):
            self.t += dt
            if self.t >= 0.6:  # 第 3 段结束时（墙钟 0.6s）中止
                raise KeyboardInterrupt

    clk = AbortClock()
    status = monitor.run_replay(
        "001", duration_sec=2.0, root=data_root, out_root=tmp_path,
        run_id="x", pacer=pace.Pacer(sleep_fn=clk.sleep, clock=clk.now),
    )
    assert status == "aborted"
    run_dir = tmp_path / "replay" / "sub-001" / "x"
    rows = _read_jsonl(run_dir / "replay.jsonl")
    assert rows[-1]["type"] != "summary"
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "aborted"


def test_preflight_failure_writes_nothing(data_root, tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        monitor.run_replay("999", root=data_root, out_root=tmp_path, run_id="z")
    assert not (tmp_path / "replay").exists()
