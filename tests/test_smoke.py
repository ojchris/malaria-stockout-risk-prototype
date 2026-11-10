import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import stockout


def test_package_imports():
    assert stockout.__version__.startswith("0.1")
