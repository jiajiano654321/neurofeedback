"""按 n-back 负荷分组、中位数、单调门控（IT2 硬判据）、出 CSV/PNG。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, io_bids
from . import theta


def median_by_load(df: pd.DataFrame) -> dict[int, float]:
    return {int(k): float(v) for k, v in df.groupby("load")["theta_uV2"].median().items()}


def gate_passes(medians: dict[int, float]) -> bool:
    """硬门控：median(L1) < L2 < L3 < L4 严格成立。任一档位缺失 → False（无法确认单调）。"""
    vals = [medians.get(i) for i in (1, 2, 3, 4)]
    if any(v is None for v in vals):
        return False
    return all(vals[i] < vals[i + 1] for i in range(3))


def _compute(subj: str, root=None) -> pd.DataFrame:
    """返回每试次 DataFrame: load, theta_uV2（已全程滤波→epoch→谱）。"""
    fz = io_bids.load_fz_uV(subj, root)
    ev = io_bids.load_events(subj, root)
    tr = io_bids.real_trial_events(ev)
    fz_f = theta.filter_whole(fz)  # 先全程滤波
    eps = theta.slice_epochs(fz_f, tr["onset"].values)  # 再切 epoch
    powers = theta.theta_power_uV2(eps)
    # 末尾不完整 epoch 可能被丢，截取与 eps 等长的试次表（按 onset 排序，丢的总是最后几个）
    n = powers.shape[0]
    return pd.DataFrame({"load": tr["nback_level"].values[:n], "theta_uV2": powers})


def run_probe(subj: str, root=None, out_dir="outputs") -> bool:
    """单被试：出 CSV+PNG，返回门控是否通过。"""
    io_bids.preflight(subj, root)
    df = _compute(subj, root)
    med = median_by_load(df)
    n_ep = {int(k): int(v) for k, v in df.groupby("load").size().items()}

    out = Path(out_dir) / "theta" / subj
    out.mkdir(parents=True, exist_ok=True)
    csv = pd.DataFrame(
        {"load": [1, 2, 3, 4],
         "n_epochs": [n_ep.get(i, 0) for i in (1, 2, 3, 4)],
         "median_theta_uV2": [med[i] for i in (1, 2, 3, 4)]}
    )
    csv.to_csv(out / "theta_by_load.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3, 4], [med[i] for i in (1, 2, 3, 4)], "o-")
    ax.set(xlabel="n-back level", ylabel="median Fz theta power (µV²)",
           title=f"sub-{subj} theta by n-back load")
    ax.set_xticks([1, 2, 3, 4])
    fig.tight_layout()
    fig.savefig(out / "theta_by_load.png", dpi=120)
    plt.close(fig)
    return gate_passes(med)


def run_all(root=None, out_dir="outputs", subs=None) -> list[dict]:
    """多被试汇总（参考用，不作硬门控）。

    逐被试容错：preflight / _compute 抛出 (FileNotFoundError, ValueError, KeyError)
    时只记录 error、不中断批处理；其它异常照常上抛。
    """
    subs = subs or config.RECOMMENDED_SUBS
    rows: list[dict] = []
    error_subs: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 5))
    for s in subs:
        try:
            io_bids.preflight(s, root)
            df = _compute(s, root)
            med = median_by_load(df)
            ok = gate_passes(med)
            rows.append({
                "subj": s, "status": "ok",
                "m1": med[1], "m2": med[2], "m3": med[3], "m4": med[4],
                "pass": ok, "error": "",
            })
            ax.plot([1, 2, 3, 4], [med[i] for i in (1, 2, 3, 4)], "o-",
                    label=f"sub-{s}{'' if ok else ' FAIL'}")
        except (FileNotFoundError, ValueError, KeyError) as e:
            rows.append({
                "subj": s, "status": "error",
                "m1": None, "m2": None, "m3": None, "m4": None,
                "pass": False, "error": str(e),
            })
            error_subs.append(s)

    out = Path(out_dir) / "theta" / "all"
    out.mkdir(parents=True, exist_ok=True)

    # CSV 列固定，失败行 m1..m4 为空
    pd.DataFrame(rows, columns=["subj", "status", "m1", "m2", "m3", "m4", "pass", "error"]
                 ).to_csv(out / "summary.csv", index=False)

    title = "theta by load — recommended subjects"
    if error_subs:
        title += f"\n({len(error_subs)} subject(s) excluded by preflight: {', '.join(error_subs)})"
    ax.set(xlabel="n-back level", ylabel="median Fz theta power (µV²)", title=title)
    ax.set_xticks([1, 2, 3, 4])
    # 有成功被试才画 legend（空 legend 会报错）
    if any(r["status"] == "ok" for r in rows):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=120)
    plt.close(fig)
    return rows
