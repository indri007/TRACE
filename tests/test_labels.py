"""Label parsing and audit — the bug this project found must never come back silently."""
import os
from pathlib import Path

import pytest

import trace_experiments as te
import trace_novelty as tn


def test_bbox_row_is_unchanged():
    assert te.row_to_xywh(["1", "0.5", "0.5", "0.2", "0.4"]) == (1, 0.5, 0.5, 0.2, 0.4)


def test_polygon_row_becomes_enclosing_box():
    c, cx, cy, w, h = te.row_to_xywh(["0", "0.1", "0.1", "0.3", "0.1", "0.3", "0.4", "0.1", "0.4"])
    assert c == 0
    assert (cx, cy, w, h) == pytest.approx((0.2, 0.25, 0.2, 0.3))


def test_polygon_is_clipped_to_image():
    _, cx, cy, w, h = te.row_to_xywh(["0", "-0.1", "0.2", "0.5", "0.2", "0.5", "1.2"])
    assert 0 <= cx - w / 2 and cx + w / 2 <= 1 and cy + h / 2 <= 1


def test_audit_finds_mixed_files(tiny_dataset):
    info = tn.audit_dataset(tiny_dataset)
    assert info["mixed_files"] == 3            # one per split
    assert info["corrupted_bbox_rows"] == 3


def test_clean_copy_is_bbox_only_and_not_resolved_back(tiny_dataset, tmp_path):
    out = tmp_path / "out"
    te.stage_audit(type("A", (), {"data_root": str(tiny_dataset)})(), out)
    clean = out / "data_clean"
    for f in clean.rglob("labels/*.txt"):
        for line in f.read_text().splitlines():
            assert len(line.split()) == 5, f"non-bbox row left in {f}"
    # Ultralytics calls Path.resolve() on the split folder: it must stay inside data_clean
    assert (clean / "train" / "images").resolve().is_relative_to(clean.resolve())
    assert all(p.is_symlink() for p in (clean / "train" / "images").iterdir())


def test_class_mapping_drops_and_remaps(tiny_dataset, tmp_path):
    tn.OUT = tmp_path
    tn.build_copy(tiny_dataset, tmp_path / "m", ["bus", "car"], ["bus", "car", "van"], {"car": "van", "bus": "drop"})
    rows = (tmp_path / "m" / "train" / "labels" / "mixed.txt").read_text().split()
    assert rows[0] == "2" and len(rows) == 5   # car -> van (id 2), bus polygon dropped


@pytest.mark.data
@pytest.mark.skipif(not Path("data/train/labels").exists(), reason="local dataset not present")
def test_real_dataset_has_no_mixed_files():
    """Guard for the real data: fails until the polygon labels are converted (use results/data_clean)."""
    root = Path(os.environ.get("TRACE_DATA", "data"))
    assert tn.audit_dataset(root)["mixed_files"] == 0
