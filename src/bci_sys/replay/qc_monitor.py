"""QC 回放主循环（IT3）：preflight → 三路流式因果带通 → 120s 校准冻结 →
逐段 L1/L3 裁决 → 终端标色 + qc.jsonl / run.json / 有效率时间轴 PNG。

A4：qc.jsonl 只含数据侧量，同参两次运行字节级一致；墙钟只进 run.json。
C5：无效段只打标并记录原因，绝不删除；反馈暂停是 IT4 的事。
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .. import config, gitmeta, io_bids
from . import causal_filter, pace, qc, qc_timeline, segments

SCHEMA_VERSION = "1.0"

GRAY = "\x1b[90m"
FZ = config.TARGET.channel
EOGS = config.EOG_PROXY_CHANNELS
ALL_CH = (FZ, *EOGS)


def _finite_or_none(v):
    """非有限值（NaN/Inf）落盘为 null：裸 NaN/Infinity 不是合法 JSON token。"""
    if v is None:
        return None
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def format_seg_line(
    seg: int, t1: float, clean_p2p: float | None, eog_p2p: float | None,
    ratio: float, verdict: dict,
) -> str:
    color = pace.rate_color(ratio)
    if verdict["phase"] == "calibration":
        clean_s = eog_s = "   —"
        mark = f"{GRAY}CAL{pace.RESET}"
    else:
        clean_s = f"{clean_p2p:6.1f}"
        eog_s = f"{eog_p2p:6.1f}"
        if verdict["valid"]:
            mark = f"{pace.GREEN}✓{pace.RESET}"
        else:
            mark = f"{pace.RED}✗ {' '.join(verdict['reasons'])}{pace.RESET}"
    return (
        f"{seg:05d} │ t={t1:6.1f}s │ Fz_clean p2p {clean_s}µV │ "
        f"EOG p2p {eog_s}µV │ rate {color}{ratio:.2f}x{pace.RESET} │ {mark}"
    )


def run_qc(
    subj: str,
    duration_sec: float = 0.0,
    root: Path | str | None = None,
    out_root: Path | str = "outputs",
    run_id: str | None = None,
    pacer: pace.Pacer | None = None,
) -> str:
    """跑一次 QC 回放。返回 'ok' / 'aborted'。preflight 失败直接抛异常（不写文件，C7）。"""
    io_bids.preflight(subj, root)  # 硬停，不写任何文件

    chans = io_bids.load_channels_uV(subj, list(ALL_CH), root)
    fz_all = chans[FZ]
    n_planned = segments.n_planned_segments(
        fz_all.size, config.SEGMENT_SAMPLES, duration_sec
    )

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

    # 参数回显放在 mkdir 之前：cp936 控制台 µ 编码崩溃时不留空目录
    print(
        f"[qc-monitor] sub-{subj} | 因果带通 "
        f"{config.FILTER_BAND_HZ[0]:g}–{config.FILTER_BAND_HZ[1]:g}Hz | "
        f"校准 {config.QC_CALIBRATION_SEC:g}s（预热剔除 {config.QC_FILTER_WARMUP_SEC:g}s）| "
        f"L1 残差 p2p>{config.QC_L1_P2P_UV:g}µV | "
        f"L3 EOG p2p>{config.QC_L3_EOG_P2P_UV:g}µV（{EOGS[0]}/{EOGS[1]} 任一）| "
        f"段 {config.SEGMENT_SEC:g}s"
    )

    run_dir = Path(out_root) / "qc" / f"sub-{subj}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pacer = pacer or pace.Pacer()
    wall_start = datetime.now().isoformat(timespec="seconds")
    pacer.start()

    band = causal_filter.CausalBandpass(ALL_CH)
    judge = qc.QcJudge()

    counters = {
        "n_running": 0, "n_valid": 0, "n_invalid": 0,
        "n_l1": 0, "n_l3": 0, "n_l1_and_l3": 0,
    }
    status = "ok"
    last_ratio = 1.0
    n_emitted = 0
    jsonl_path = run_dir / "qc.jsonl"

    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "code_version": gitmeta.code_version(),
            "subj": subj,
            "sfreq": config.SFREQ,
            "segment_sec": config.SEGMENT_SEC,
            "channels": {FZ: FZ, EOGS[0]: EOGS[0], EOGS[1]: EOGS[1]},
            "filter_band_hz": list(config.FILTER_BAND_HZ),
            "calibration_sec": config.QC_CALIBRATION_SEC,
            "filter_warmup_sec": config.QC_FILTER_WARMUP_SEC,
            "l1_p2p_uv": config.QC_L1_P2P_UV,
            "l3_eog_p2p_uv": config.QC_L3_EOG_P2P_UV,
            "n_segments_planned": n_planned,
        }, ensure_ascii=False) + "\n")

        try:
            for seg in segments.iter_segments(
                fz_all, config.SFREQ, config.SEGMENT_SAMPLES, duration_sec
            ):
                s = seg.seg * config.SEGMENT_SAMPLES
                e = s + config.SEGMENT_SAMPLES
                filtered = band.push_block({c: chans[c][s:e] for c in ALL_CH})
                verdict = judge.push(seg.t1, filtered)

                if verdict.get("calib") is not None:
                    f.write(json.dumps(verdict["calib"], ensure_ascii=False) + "\n")
                    print(
                        f"[校准] b_{EOGS[0]}={judge.b_fp1:.3f} "
                        f"b_{EOGS[1]}={judge.b_fp2:.3f} (frozen)"
                    )

                f.write(json.dumps({
                    "type": "seg",
                    "seg": seg.seg,
                    "t0": seg.t0,
                    "t1": seg.t1,
                    "phase": verdict["phase"],
                    "fz_clean_p2p": _finite_or_none(verdict["fz_clean_p2p"]),
                    "eog_p2p": _finite_or_none(verdict["eog_p2p"]),
                    "valid": verdict["valid"],
                    "reasons": verdict["reasons"],
                }, ensure_ascii=False) + "\n")

                # 计数必须在 wait 之前：Ctrl-C 几乎总落在 wait 窗口，
                # jsonl 已写入的段数必须与 run.json n_segments 一致。
                n_emitted += 1

                if verdict["phase"] == "running":
                    counters["n_running"] += 1
                    hit_l1 = "L1" in verdict["reasons"]
                    hit_l3 = "L3" in verdict["reasons"]
                    if verdict["valid"]:
                        counters["n_valid"] += 1
                    else:
                        counters["n_invalid"] += 1
                    if hit_l1:
                        counters["n_l1"] += 1
                    if hit_l3:
                        counters["n_l3"] += 1
                    if hit_l1 and hit_l3:
                        counters["n_l1_and_l3"] += 1

                ratio = pacer.rate_ratio(seg.t1)
                last_ratio = ratio
                print(format_seg_line(
                    seg.seg, seg.t1,
                    verdict["fz_clean_p2p"], verdict["eog_p2p"],
                    ratio, verdict,
                ))

                pacer.wait_after_segment(seg.seg)
        except KeyboardInterrupt:
            status = "aborted"

        if status == "ok":
            valid_rate = (
                None if counters["n_running"] == 0
                else round(counters["n_valid"] / counters["n_running"], 6)
            )
            f.write(json.dumps({
                "type": "summary",
                **counters,
                "valid_rate": valid_rate,
            }, ensure_ascii=False) + "\n")

    # 出图先于 run.json：绘图失败硬抛、无 run.json，
    # 不留 status=ok 但缺 PNG 的矛盾目录。
    if status == "ok":
        qc_timeline.plot_timeline(
            jsonl_path, run_dir / "valid_rate_timeline.png", subj=subj
        )

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
