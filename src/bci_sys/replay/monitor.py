"""回放主循环：preflight → 分段 1x 回放 → 终端显示 + JSONL/run.json 落盘。

确定性（A4）：replay.jsonl 只含数据侧量，两次同参运行字节级一致；
墙钟量隔离到 run.json。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import config, gitmeta, io_bids
from . import pace, segments

SCHEMA_VERSION = "1.0"


def _jsonable(v):
    """pandas/NA → JSON 可序列化。"""
    if v is None or pd.isna(v):
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _event_kind(row) -> str:
    if row["trial_type"] == "dropped_samples":
        return "dropped"
    nb = row.get("nback_level")
    if pd.notna(nb) and int(nb) in (1, 2, 3, 4):
        return "trial"
    return "marker"


def _event_line(t: float, row) -> dict:
    nb = row.get("nback_level")
    return {
        "type": "event",
        "t": float(t),
        "trial_type": _jsonable(row["trial_type"]),
        "nback_level": int(nb) if pd.notna(nb) else None,
        "istutorial": bool(row["istutorial"]) if pd.notna(row.get("istutorial")) else None,
        "kind": _event_kind(row),
    }


def format_seg_line(seg: int, t1: float, p2p: float, ratio: float, marks: str) -> str:
    color = pace.rate_color(ratio)
    return (
        f"{seg:05d} │ t={t1:6.1f}s │ Fz p2p {p2p:6.1f}µV │ "
        f"rate {color}{ratio:.2f}x{pace.RESET} │ {marks}"
    )


def _event_marks(evs: pd.DataFrame) -> str:
    parts = []
    for _, r in evs.iterrows():
        if _event_kind(r) == "dropped":
            parts.append(f"{pace.RED}✗ dropped{pace.RESET}")
        elif _event_kind(r) == "trial":
            parts.append(f"◆ {int(r['nback_level'])}-back L{int(r['nback_level'])}")
    return " ".join(parts)


def run_replay(
    subj: str,
    duration_sec: float = 0.0,
    root: Path | str | None = None,
    out_root: Path | str = "outputs",
    run_id: str | None = None,
    pacer: pace.Pacer | None = None,
) -> str:
    """跑一次回放。返回状态 'ok' / 'aborted'。preflight 失败直接抛异常（不写文件）。"""
    io_bids.preflight(subj, root)  # 硬停，不写任何文件

    fz = io_bids.load_fz_uV(subj, root)
    events = io_bids.load_events(subj, root)
    n_planned = segments.n_planned_segments(fz.size, config.SEGMENT_SAMPLES, duration_sec)

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / "replay" / f"sub-{subj}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pacer = pacer or pace.Pacer()
    wall_start = datetime.now().isoformat(timespec="seconds")
    pacer.start()

    n_events = 0
    n_dropped = 0
    jsonl_path = run_dir / "replay.jsonl"
    status = "ok"
    last_ratio = 1.0
    n_emitted = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "code_version": gitmeta.code_version(),
            "subj": subj,
            "data_file": io_bids.vhdr_path(subj, root).name,
            "sfreq": config.SFREQ,
            "segment_sec": config.SEGMENT_SEC,
            "fz_channel": config.TARGET.channel,
            "n_segments_planned": n_planned,
        }, ensure_ascii=False) + "\n")

        try:
            for seg in segments.iter_segments(
                fz, config.SFREQ, config.SEGMENT_SAMPLES, duration_sec
            ):
                f.write(json.dumps({
                    "type": "seg",
                    "seg": seg.seg,
                    "t0": seg.t0,
                    "t1": seg.t1,
                    "fz_uV": [float(v) for v in seg.samples],
                }, ensure_ascii=False) + "\n")

                evs = segments.segment_events(events, seg.t0, seg.t1)
                for _, row in evs.iterrows():
                    line = _event_line(row["onset"], row)
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    n_events += 1
                    if line["kind"] == "dropped":
                        n_dropped += 1

                p2p = float(seg.samples.max() - seg.samples.min())
                ratio = pacer.rate_ratio(seg.t1)
                last_ratio = ratio
                print(format_seg_line(seg.seg, seg.t1, p2p, ratio, _event_marks(evs)))

                pacer.wait_after_segment(seg.seg)
                n_emitted += 1
        except KeyboardInterrupt:
            status = "aborted"

        if status == "ok":
            f.write(json.dumps({
                "type": "summary",
                "n_segments": n_emitted,
                "data_elapsed_s": n_emitted * config.SEGMENT_SEC,
                "tail_s": segments.tail_seconds(fz.size, duration_sec=duration_sec),
                "n_events": n_events,
                "n_dropped": n_dropped,
            }, ensure_ascii=False) + "\n")

    run_json = {
        "subj": subj,
        "status": status,
        "wall_started_iso": wall_start,
        "wall_elapsed_s": round(pacer.wall_elapsed(), 3),
        "rate_ratio": round(last_ratio, 4),
        "n_segments": n_emitted,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status
