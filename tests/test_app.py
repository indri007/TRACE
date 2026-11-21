"""Application contract: config, webhook payload, model loading and inference."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_config_classes_match_dataset():
    import config
    assert config.CLASS_NAMES == ["bus", "car", "van"]
    assert config.IMG_SIZE == 640


def test_payload_alert_threshold_boundary(monkeypatch):
    from utils import n8n_client
    monkeypatch.setattr(n8n_client, "ALERT_THRESHOLD", 10)
    det = [{"confidence": 0.8}]
    at = n8n_client.build_payload("a.jpg", {"car": 10, "van": 0, "bus": 0}, det)
    over = n8n_client.build_payload("a.jpg", {"car": 11, "van": 0, "bus": 0}, det)
    assert at["alert"] is False and over["alert"] is True     # alert only when total > threshold
    assert set(at) >= {"timestamp", "image_name", "detections", "total_vehicles", "avg_confidence", "alert"}


def test_payload_without_detections():
    from utils import n8n_client
    p = n8n_client.build_payload("empty.jpg", {"car": 0, "van": 0, "bus": 0}, [])
    assert p["total_vehicles"] == 0 and p["avg_confidence"] == 0.0


def test_send_handles_unreachable_webhook(monkeypatch):
    from utils import n8n_client
    monkeypatch.setattr(n8n_client, "N8N_WEBHOOK_URL", "http://127.0.0.1:9/none")
    monkeypatch.setattr(n8n_client, "N8N_TIMEOUT_SEC", 1)
    ok, msg = n8n_client.send_to_n8n("a.jpg", {"car": 1, "van": 0, "bus": 0}, [{"confidence": 0.9}])
    assert ok is False and msg                 # the app must not crash when n8n is down


@pytest.mark.slow
def test_deployed_model_loads_with_expected_classes():
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    w = ROOT / "app" / "models" / "best.pt"
    if not w.exists():
        pytest.skip("best.pt not present")
    m = YOLO(str(w))
    assert [m.names[i] for i in sorted(m.names)] == ["bus", "car", "van"]


@pytest.mark.slow
def test_inference_smoke_on_cpu():
    pytest.importorskip("ultralytics")
    import numpy as np
    from ultralytics import YOLO
    w = ROOT / "app" / "models" / "best.pt"
    if not w.exists():
        pytest.skip("best.pt not present")
    img = (np.random.default_rng(0).random((640, 640, 3)) * 255).astype("uint8")
    r = YOLO(str(w)).predict(img, imgsz=640, conf=0.40, device="cpu", verbose=False)[0]
    assert r.boxes is not None and r.orig_shape == (640, 640)
