"""IT4 前端静态契约（视觉行为由 T9 人工回放截图验收，不引入浏览器测试栈）。"""
from pathlib import Path

from fastapi.testclient import TestClient

from bci_sys.replay import webapp

WEB = Path(__file__).resolve().parents[1] / "src" / "bci_sys" / "web"


def test_index_contains_all_eight_phase_screens_and_controls():
    html = (WEB / "index.html").read_text("utf-8").lower()
    for sid in ("screen-confirm", "screen-calibration", "screen-freeze",
                "screen-block", "screen-rest", "screen-finished", "screen-aborted",
                "replay-bar"):
        assert f'id="{sid}"' in html
    for token in ("btn-start", "btn-abort", "cal-dots", "block-dots",
                  "hit-bar", "hit-tick", "theta-gauge", "theta-thr",
                  "invalid-banner", "rest-countdown", "rates-body",
                  "replay-seek", "btn-replay-play"):
        assert token in html
    assert 'src="/app.js"' in html
    assert "http://" not in html and "https://" not in html  # 无外链/CDN


def test_app_js_protocol_contract():
    js = (WEB / "app.js").read_text("utf-8")
    for token in ("/ws", "/api/start", "/api/abort", "/api/replay",
                  '"snapshot"', "calib_tick", "calib_seg", "block_start",
                  "rest_start", "block_summary", "final", "aborted",
                  "lastSeq", "boundaries"):
        assert token in js, token
    assert "http://" not in js and "https://" not in js
    # 关页/断流不允许调用 abort
    assert "beforeunload" not in js and "onbeforeunload" not in js


def test_static_served_via_app():
    with TestClient(webapp.create_app()) as c:
        assert c.get("/").headers["content-type"].startswith("text/html")
        assert "javascript" in c.get("/app.js").headers["content-type"]
