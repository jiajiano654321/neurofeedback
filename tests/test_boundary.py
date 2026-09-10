import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "bci_sys"


def _imports_of(pkg: pathlib.Path) -> set[str]:
    mods = set()
    for f in pkg.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module)
            elif isinstance(n, ast.Import):
                mods.update(a.name for a in n.names)
    return mods


def test_replay_does_not_import_offline():
    """C3：实时路径不得 import 离线路径。"""
    mods = _imports_of(SRC / "replay")
    assert not any(m == "bci_sys.offline" or m.startswith("bci_sys.offline.")
                   for m in mods), f"replay 违规 import offline: {mods}"


def test_offline_does_not_import_replay():
    """C3：离线路径不得 import 实时路径。"""
    mods = _imports_of(SRC / "offline")
    assert not any(m == "bci_sys.replay" or m.startswith("bci_sys.replay.")
                   for m in mods), f"offline 违规 import replay: {mods}"


def test_replay_never_calls_filtfilt():
    """C3：实时路径禁止零相位滤波（filtfilt 用未来数据），全目录 AST 守卫。"""
    offenders = []
    for f in (SRC / "replay").rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        # 只禁【调用/引用】这个属性名；注释/docstring 经 AST 解析天然不算
        if "filtfilt" in attrs:
            offenders.append(f.name)
    assert not offenders, f"replay 模块出现 filtfilt: {offenders}"