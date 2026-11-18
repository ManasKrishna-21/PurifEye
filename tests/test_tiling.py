"""Unit tests for tiling coordinate round-trips, coverage, and fragment mapping."""

import pytest
from purifeye.tiling import generate_tile_coordinates, map_box_to_tile, map_tile_detections_to_full_image
from purifeye.merging import compute_iou, class_aware_nms


def test_tiling_full_coverage():
    w, h = 1920, 1080
    tiles = generate_tile_coordinates(w, h, tile_size=640, overlap=0.20)
    assert len(tiles) > 0

    # Ensure all corners are covered
    covers_x0 = any(t[0] == 0 for t in tiles)
    covers_y0 = any(t[1] == 0 for t in tiles)
    covers_xw = any(t[2] == w for t in tiles)
    covers_yh = any(t[3] == h for t in tiles)

    assert covers_x0 and covers_y0 and covers_xw and covers_yh


def test_box_to_tile_mapping_and_reprojection():
    tile = (100, 100, 740, 740)
    # Box completely inside tile
    box = (200, 200, 300, 300)
    res = map_box_to_tile(box, tile, min_visibility=0.40)
    assert res is not None

    # Partial box with less than 40% visibility
    box_barely_visible = (50, 50, 110, 110)
    res_barely = map_box_to_tile(box_barely_visible, tile, min_visibility=0.40)
    assert res_barely is None


def test_tile_detection_shift():
    tile = (200, 300, 840, 940)
    tile_dets = [{"box": [10, 20, 50, 60], "confidence": 0.8, "class_id": 0}]
    mapped = map_tile_detections_to_full_image(tile_dets, tile)

    assert mapped[0]["box"] == [210, 320, 250, 360]


def test_class_aware_nms():
    dets = [
        {"box": [100, 100, 200, 200], "confidence": 0.9, "class_id": 0},
        {"box": [105, 102, 202, 198], "confidence": 0.7, "class_id": 0},
        {"box": [500, 500, 600, 600], "confidence": 0.85, "class_id": 0},
    ]
    kept = class_aware_nms(dets, iou_threshold=0.45, conf_threshold=0.25)
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.9
    assert kept[1]["confidence"] == 0.85
