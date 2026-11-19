"""Class-aware Non-Maximum Suppression (NMS) and tile fragment merging."""

from typing import List, Dict, Any
import numpy as np


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def class_aware_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25,
    merge_fragments: bool = True
) -> List[Dict[str, Any]]:
    """Perform class-aware Non-Maximum Suppression across detections.

    Each detection is a dict with keys:
    - 'box': [x1, y1, x2, y2]
    - 'confidence': float
    - 'class_id': int
    """
    filtered = [d for d in detections if d["confidence"] >= conf_threshold]
    if not filtered:
        return []

    # Group by class_id
    classes = set(d["class_id"] for d in filtered)
    final_keep = []

    for c in classes:
        cls_dets = [d for d in filtered if d["class_id"] == c]
        cls_dets.sort(key=lambda x: x["confidence"], reverse=True)

        keep = []
        suppressed = [False] * len(cls_dets)

        for i in range(len(cls_dets)):
            if suppressed[i]:
                continue
            cur = cls_dets[i]
            keep_box = list(cur["box"])

            for j in range(i + 1, len(cls_dets)):
                if suppressed[j]:
                    continue
                other = cls_dets[j]
                iou = compute_iou(keep_box, other["box"])

                if iou >= iou_threshold:
                    if merge_fragments:
                        # Soft expansion for cut-off bottles across tile boundaries
                        keep_box[0] = min(keep_box[0], other["box"][0])
                        keep_box[1] = min(keep_box[1], other["box"][1])
                        keep_box[2] = max(keep_box[2], other["box"][2])
                        keep_box[3] = max(keep_box[3], other["box"][3])
                    suppressed[j] = True

            cur_copy = cur.copy()
            cur_copy["box"] = keep_box
            keep.append(cur_copy)

        final_keep.extend(keep)

    final_keep.sort(key=lambda x: x["confidence"], reverse=True)
    return final_keep
