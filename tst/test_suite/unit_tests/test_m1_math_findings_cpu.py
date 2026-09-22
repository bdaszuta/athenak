"""Run the native M1 math regressions in the standard CPU suite."""
import sys
from pathlib import Path

TST_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TST_DIR))
from test_m1_math_findings import run  # noqa: E402


def test_m1_math_findings_cpu():
    run(TST_DIR / "build")
