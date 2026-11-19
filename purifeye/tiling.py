"""Overlapped tiling and coordinate transformation logic for high-resolution images."""

from typing import List, Tuple
import numpy as np


def generate_tile_coordinates(img_w: int, img_h: int, tile_size: int = 640, overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Generate (xmin, ymin, xmax, ymax) coordinates for tiles covering the entire image."""
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0, img_w, img_h)]

    step = max(1, int(tile_size * (1.0 - overlap)))
    x_starts = list(range(0, max(1, img_w - tile_size + 1), step))
    if not x_starts or x_starts[-1] + tile_size < img_w:
        x_starts.append(max(0, img_w - tile_size))

    y_starts = list(range(0, max(1, img_h - tile_size + 1), step))
    if not y_starts or y_starts[-1] + tile_size < img_h:
        y_starts.append(max(0, img_h - tile_size))

    tiles = []
    for ys in y_starts:
        for xs in x_starts:
            xe = min(img_w, xs + tile_size)
            ye = min(img_h, ys + tile_size)
            tiles.append((xs, ys, xe, ye))
    return tiles


def map_box_to_tile(
    box: Tuple[float, float, float, float],
    tile: Tuple[int, int, int, int],
    min_visibility: float = 0.40
) -> Tuple[float, float, float, float] | None:
    """Map absolute box [xmin, ymin, xmax, ymax] into tile-relative normalized [xc, yc, w, h].

    Returns None if the visible fraction in the tile is less than min_visibility.
    """
    bx1, by1, bx2, by2 = box
    tx1, ty1, tx2, ty2 = tile
    tile_w = tx2 - tx1
    tile_h = ty2 - ty1

    box_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    if box_area <= 0:
        return None

    # Intersection
    ix1 = max(bx1, tx1)
    iy1 = max(by1, ty1)
    ix2 = min(bx2, tx2)
    iy2 = min(by2, ty2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if (inter_area / box_area) < min_visibility:
        return None

    # Tile-relative normalized coordinates
    rel_xc = ((ix1 + ix2) / 2.0 - tx1) / tile_w
    rel_yc = ((iy1 + iy2) / 2.0 - ty1) / tile_h
    rel_w = inter_w / tile_w
    rel_h = inter_h / tile_h

    # Clamping
    rel_xc = min(max(rel_xc, 0.0), 1.0)
    rel_yc = min(max(rel_yc, 0.0), 1.0)
    rel_w = min(max(rel_w, 0.0), 1.0)
    rel_h = min(max(rel_h, 0.0), 1.0)

    return (rel_xc, rel_yc, rel_w, rel_h)


def map_tile_detections_to_full_image(
    detections: List[dict],
    tile: Tuple[int, int, int, int]
) -> List[dict]:
    """Shift tile-space detections back to full-image coordinate space.

    detections: list of dicts with 'box': [xmin, ymin, xmax, ymax] (in tile pixels), 'confidence', 'class_id'.
    """
    tx1, ty1, _, _ = tile
    mapped = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        item = d.copy()
        item["box"] = [x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1]
        mapped.append(item)
    return mapped
