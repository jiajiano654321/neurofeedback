import pytest


@pytest.fixture
def data_root():
    """返回数据根目录；ds007169 不在则 skip（数据 gitignore，不进仓库）。"""
    from bci_sys import config

    p = config.data_root()
    if not (p / "sub-001" / "eeg").exists():
        pytest.skip("ds007169 data not available")
    return p
