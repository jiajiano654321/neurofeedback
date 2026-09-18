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


def test_app_js_final_state_refreshes_run_info_and_rerenders():
    """终态（final/aborted）后必须重拉 /api/run 并重新渲染，
    否则 finished 屏墙钟永远是 boot 时的旧值（—s）。"""
    import re
    js = (WEB / "app.js").read_text("utf-8")
    # 存在异步刷新函数，内部 fetch /api/run
    assert "async function fetchRunInfo" in js
    assert 'fetch("/api/run")' in js
    # final / aborted 分支都调用它（live 模式）
    assert "fetchRunInfo()" in js
    # fetch 完成后必须触发 render() —— 否则 S.runInfo 更新了但 finished 屏不重画
    # 提取 fetchRunInfo 声明到下一个顶层 function 声明之间的区间，
    # 避免被内部 try/catch 的 } 误截断
    fn_start = js.index("async function fetchRunInfo")
    rest = js[fn_start:]
    next_fn = re.search(r"\nfunction \w", rest[1:])
    if next_fn:
        fn_body = rest[:next_fn.start() + 1]
    else:
        fn_body = rest
    assert "render()" in fn_body, "fetchRunInfo 完成后必须调用 render() 重绘 finished 屏"


def test_static_served_via_app():
    with TestClient(webapp.create_app()) as c:
        assert c.get("/").headers["content-type"].startswith("text/html")
        assert "javascript" in c.get("/app.js").headers["content-type"]
