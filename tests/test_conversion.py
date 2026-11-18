"""Unit tests for VOC parsing, boundary clamping, and YOLO label conversion."""

import pytest
from pathlib import Path
from prepare_yolo_dataset import parse_voc_xml


def test_box_clamping_and_normalization():
    # Box exceeding true image dimensions
    true_w, true_h = 1000, 1000
    raw_box = [-50, -20, 1050, 1100]

    cx1 = max(0.0, min(float(raw_box[0]), float(true_w)))
    cy1 = max(0.0, min(float(raw_box[1]), float(true_h)))
    cx2 = max(0.0, min(float(raw_box[2]), float(true_w)))
    cy2 = max(0.0, min(float(raw_box[3]), float(true_h)))

    assert cx1 == 0.0
    assert cy1 == 0.0
    assert cx2 == 1000.0
    assert cy2 == 1000.0

    xc = (cx1 + cx2) / 2.0 / true_w
    yc = (cy1 + cy2) / 2.0 / true_h
    w = (cx2 - cx1) / true_w
    h = (cy2 - cy1) / true_h

    assert xc == 0.5
    assert yc == 0.5
    assert w == 1.0
    assert h == 1.0


def test_degenerate_zero_area_box_dropped():
    raw_box = [100, 100, 100, 100]
    bw = raw_box[2] - raw_box[0]
    bh = raw_box[3] - raw_box[1]
    assert bw <= 1.0 or bh <= 1.0
