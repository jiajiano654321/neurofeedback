"""事后时间轴图：读合成 qc.jsonl 标签出 PNG；不参与裁决。"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    specs = []
    for k in range(600):  # 0–120s 校准
        specs.append(((k + 1) * 0.2, "calibration", None, []))
    for k in range(600, 750):  # 120–150s：交替有效/无效
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


def test_timeline_all_calibration_still_renders(tmp_path):
    specs = [((k + 1) * 0.2, "calibration", None, []) for k in range(10)]
    jsonl = tmp_path / "qc.jsonl"
    png = tmp_path / "valid_rate_timeline.png"
    _write_jsonl(jsonl, specs)
    qc_timeline.plot_timeline(jsonl, png, subj="001")  # 不抛错即通过
    assert png.exists()
