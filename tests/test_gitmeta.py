import re

from bci_sys import gitmeta


def test_code_version_is_hash_or_unknown():
    v = gitmeta.code_version()
    # 仓库内：7+ 位 hash，可能带 -dirty；仓库外：'unknown'
    assert v == "unknown" or re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", v)
