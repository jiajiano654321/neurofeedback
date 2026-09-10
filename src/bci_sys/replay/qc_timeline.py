"""事后绘图：读 qc.jsonl 的因果标签画有效率时间轴（IT3 产物）。

只在回放结束后运行；输入是已经产生的标签，输出是 PNG，
不向 qc.py 的任何裁决回流信息（C3：事后处理与实时因果判定隔离）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

BIN_SEC = 30.0
COLORS = {
    "calibration": "#bdbdbd",  # 灰
    "valid": "#2ca02c",        # 绿
    "L1": "#ff7f0e",           # 橙：仅 L1
    "L3": "#d62728",           # 红：仅 L3
    "both": "#7b1fa2",         # 紫：L1+L3
}


def load_seg_rows(qc_jsonl_path: Path | str) -> list[dict]:
    rows = []
    for line in Path(qc_jsonl_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("type") == "seg":
            rows.append(rec)
    return rows


def _row_color(row: dict) -> str:
    if row["phase"] == "calibration":
        return COLORS["calibration"]
    if row["valid"]:
        return COLORS["valid"]
    has_l1 = "L1" in row["reasons"]
    has_l3 = "L3" in row["reasons"]
    if has_l1 and has_l3:
        return COLORS["both"]
    return COLORS["L1"] if has_l1 else COLORS["L3"]


def _bin_valid_rates(rows: list[dict], bin_sec: float = BIN_SEC):
    """仅运行期段按 t1 落 30s 箱；返回 (箱中心秒, 有效率)。校准箱自然不出现。"""
    run = [r for r in rows if r["phase"] == "running"]
    if not run:
        return np.array([]), np.array([])
    t1 = np.array([r["t1"] for r in run], dtype=float)
    valid = np.array([1.0 if r["valid"] else 0.0 for r in run])
    b = np.floor((t1 - 1e-9) / bin_sec).astype(int)
    xs, ys = [], []
    for k in range(b.min(), b.max() + 1):
        m = b == k
        if m.any():
            xs.append((k + 0.5) * bin_sec)
            ys.append(float(valid[m].mean()))
    return np.array(xs), np.array(ys)


def plot_timeline(qc_jsonl_path: Path | str, png_path: Path | str, subj: str = "") -> None:
    rows = load_seg_rows(qc_jsonl_path)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 5]},
    )

    # 上轨：逐段裁决色带
    t_max = rows[-1]["t1"] if rows else 1.0
    rgb = np.array([[mcolors.to_rgb(_row_color(r)) for r in rows]])
    ax0.imshow(rgb, aspect="auto", extent=[0.0, t_max, 0.0, 1.0])
    ax0.set_yticks([])
    ax0.set_ylabel("verdict")

    # 下轨：运行期 30s 分箱有效率
    bx, by = _bin_valid_rates(rows)
    if bx.size:
        ax1.plot(bx, by, "o-", ms=4, color="#1f77b4")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("valid rate (30s bin)")
    ax1.set_xlabel("time (s)")
    ax1.grid(alpha=0.3)

    run = [r for r in rows if r["phase"] == "running"]
    title = f"sub-{subj} QC valid-rate timeline" if subj else "QC valid-rate timeline"
    if run:
        overall = sum(1 for r in run if r["valid"]) / len(run)
        title += f" — overall {overall:.1%} (running only, calibration excluded)"
    fig.suptitle(title, fontsize=11)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["calibration"], label="calibration"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["valid"], label="valid"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["L1"], label="invalid L1"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["L3"], label="invalid L3"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS["both"], label="invalid L1+L3"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=5, fontsize=7, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
