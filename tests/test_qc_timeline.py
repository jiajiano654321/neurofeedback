"""事后时间轴图：读合成 qc.jsonl 标签出 PNG；不参与裁决。"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bci_sys.replay import qc_timeline


def _write_jsonl(path, seg_specs):
    # seg_specs: [(t1, phase, valid, reasons)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "subj": "001"}) + "\n")
        for seg, (t1, phase, valid, reasons) in enumerate(seg_specs):
            f.write(json.dumps({
                "type": "seg", "seg": seg, "t0": t1 - 0.2, "t1": t1,
                "phase": phase, "fz_clean_p2p": None, "eog_p2p": None,
                "valid": valid, "reasons": reasons,
            }) + "\n")
        f.write(json.dumps({"type": "summary", "n_running": 1}) + "\n")


def test_timeline_png_from_mixed_labels(tmp_path):
    # 600 校准段（0–120s）+ 300 运行段（120.2–180.0s，跨两个 30s 箱）
    specs = []
    for k in range(600):
        specs.append(((k + 1) * 0.2, "calibration", None, []))
    for k in range(600, 900):
        t1 = (k + 1) * 0.2
        if k % 4 == 0:
            specs.append((t1, "running", False, ["L3"]))
        elif k % 4 == 1:
            specs.append((t1, "running", False, ["L1", "L3"]))
        else:
            specs.append((t1, "running", True, []))
    jsonl = tmp_path / "qc.jsonl"
    png = tmp_path / "valid_rate_timeline.png"
    _write_jsonl(jsonl, specs)

    qc_timeline.plot_timeline(jsonl, png, subj="001")

    assert png.exists() and png.stat().st_size > 1000
    img = plt.imread(png)
    assert img.ndim == 3 and img.shape[2] in (3, 4)

    # 30s 分箱数值：bin4(k=600..749) 74/150 有效，bin5(k=750..899) 76/150 有效
    rows = qc_timeline.load_seg_rows(jsonl)
    bx, by = qc_timeline._bin_valid_rates(rows)
    np.testing.assert_allclose(bx, [135.0, 165.0])
    np.testing.assert_allclose(by, [74 / 150, 76 / 150])

    # 五态配色直取
    assert qc_timeline._row_color({"phase": "calibration"}) == qc_timeline.COLORS["calibration"]
    assert qc_timeline._row_color(
        {"phase": "running", "valid": True, "reasons": []}) == qc_timeline.COLORS["valid"]
    assert qc_timeline._row_color(
        {"phase": "running", "valid": False, "reasons": ["L1"]}) == qc_timeline.COLORS["L1"]
    assert qc_timeline._row_color(
        {"phase": "running", "valid": False, "reasons": ["L3"]}) == qc_timeline.COLORS["L3"]
    assert qc_timeline._row_color(
        {"phase": "running", "valid": False, "reasons": ["L1", "L3"]}) == qc_timeline.COLORS["both"]

    # 标题含总体有效率（150/300 = 50.0%，仅运行期）；标题与图例几何不重叠
    fig, title_art, legend = qc_timeline.build_timeline_figure(rows, subj="001")
    try:
        assert "50.0%" in title_art.get_text()
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        assert not legend.get_window_extent(rend).overlaps(
            title_art.get_window_extent(rend))
    finally:
        plt.close(fig)


def test_bin_rates_nan_for_empty_bin():
    # bin4 两段 + bin6 一段，中间 bin5 空缺 → 该箱 NaN，折线断开不桥接
    rows = [
        {"type": "seg", "phase": "running", "t1": 120.2, "valid": True, "reasons": []},
        {"type": "seg", "phase": "running", "t1": 120.4, "valid": False, "reasons": ["L3"]},
        {"type": "seg", "phase": "running", "t1": 181.0, "valid": True, "reasons": []},
    ]
    bx, by = qc_timeline._bin_valid_rates(rows)
    np.testing.assert_allclose(bx, [135.0, 165.0, 195.0])
    assert by[0] == 0.5 and np.isnan(by[1]) and by[2] == 1.0


def test_timeline_all_calibration_still_renders(tmp_path):
    specs = [((k + 1) * 0.2, "calibration", None, []) for k in range(10)]
    jsonl = tmp_path / "qc.jsonl"
    png = tmp_path / "valid_rate_timeline.png"
    _write_jsonl(jsonl, specs)
    qc_timeline.plot_timeline(jsonl, png, subj="001")  # 不抛错即通过
    assert png.exists()
