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