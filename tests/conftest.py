"""Shared fixtures for the TRACE test harness."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # trace_experiments.py, trace_novelty.py
sys.path.insert(0, str(ROOT / "app"))   # config.py, utils/


@pytest.fixture
def tiny_dataset(tmp_path):
    """Three-split YOLO dataset with one clean file, one mixed bbox/polygon file and one empty file."""
    root = tmp_path / "ds"
    for s in ["train", "valid", "test"]:
        (root / s / "images").mkdir(parents=True)
        (root / s / "labels").mkdir(parents=True)
        for name, rows in {
            "clean": ["1 0.5 0.5 0.2 0.2"],
            "mixed": ["1 0.5 0.5 0.2 0.2", "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4"],
            "empty": [],
        }.items():
            (root / s / "images" / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")
            (root / s / "labels" / f"{name}.txt").write_text("\n".join(rows))
    return root
