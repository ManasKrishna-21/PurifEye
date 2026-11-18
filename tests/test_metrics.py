"""Unit tests for metric computations: IoU, mAP, and F1 optimization."""

import numpy as np
import pytest
from purifeye.metrics import compute_iou_matrix, compute_ap, evaluate_detections
from purifeye.config import load_config


def test_iou_matrix_calculation():
    b1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b2 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)

    ious = compute_iou_matrix(b1, b2)
    assert ious[0, 0] == pytest.approx(1.0, 1e-4)
    assert ious[0, 1] == pytest.approx(0.0, 1e-4)


def test_perfect_prediction_evaluation():
    gts = [
        {"image_id": "img1", "box": [10, 10, 50, 50], "class_id": 0},
        {"image_id": "img1", "box": [60, 60, 100, 100], "class_id": 0},
    ]
    preds = [
        {"image_id": "img1", "box": [10, 10, 50, 50], "confidence": 0.95, "class_id": 0},
        {"image_id": "img1", "box": [60, 60, 100, 100], "confidence": 0.90, "class_id": 0},
    ]
    res = evaluate_detections(preds, gts)
    assert res["map50"] == pytest.approx(1.0, 1e-3)
    assert res["recall"] == pytest.approx(1.0, 1e-3)
    assert res["precision"] == pytest.approx(1.0, 1e-3)


def test_config_loader():
    cfg = load_config()
    assert "hardware" in cfg
    assert "dataset" in cfg
    assert "training" in cfg
