"""Persistent object tracking and river flow counting."""

from typing import List, Dict, Any, Tuple
import numpy as np


class CentroidTracker:
    """Lightweight Centroid & IoU tracker for real-time video streams."""

    def __init__(self, max_disappeared: int = 25, iou_thresh: float = 0.30):
        self.next_object_id = 1
        self.objects: Dict[int, List[float]] = {}  # id -> [x1, y1, x2, y2]
        self.disappeared: Dict[int, int] = {}     # id -> missed frame count
        self.max_disappeared = max_disappeared
        self.iou_thresh = iou_thresh
        self.counted_ids: set = set()
        self.total_bottles_counted = 0

    def _box_centroid(self, box: List[float]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _box_iou(self, b1: List[float], b2: List[float]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / max(1e-6, union)

    def update(
        self,
        detections: List[Dict[str, Any]],
        counting_line_y: float | None = None
    ) -> List[Dict[str, Any]]:
        """Update tracked objects with new frame detections.

        detections: list of dicts with 'box' [x1, y1, x2, y2], 'confidence', 'class_id'.
        Returns list of detections with 'track_id' added.
        """
        if len(detections) == 0:
            # Mark existing objects as disappeared
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return []

        input_boxes = [d["box"] for d in detections]

        if len(self.objects) == 0:
            # Register all
            tracked_results = []
            for d in detections:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.objects[obj_id] = d["box"]
                self.disappeared[obj_id] = 0
                item = d.copy()
                item["track_id"] = obj_id
                tracked_results.append(item)
                self._check_counting_line(obj_id, d["box"], counting_line_y)
            return tracked_results

        # Match existing objects with input boxes using IoU
        object_ids = list(self.objects.keys())
        prev_boxes = [self.objects[oid] for oid in object_ids]

        iou_matrix = np.zeros((len(object_ids), len(input_boxes)), dtype=np.float32)
        for i, pb in enumerate(prev_boxes):
            for j, ib in enumerate(input_boxes):
                iou_matrix[i, j] = self._box_iou(pb, ib)

        matched_obj_indices = set()
        matched_box_indices = set()

        if iou_matrix.size > 0:
            rows = iou_matrix.max(axis=1).argsort()[::-1]
            for r in rows:
                c = int(np.argmax(iou_matrix[r]))
                if c in matched_box_indices:
                    continue
                if iou_matrix[r, c] >= self.iou_thresh:
                    matched_obj_indices.add(r)
                    matched_box_indices.add(c)
                    oid = object_ids[r]
                    self.objects[oid] = input_boxes[c]
                    self.disappeared[oid] = 0
                    self._check_counting_line(oid, input_boxes[c], counting_line_y)

        # Unmatched existing
        for r in range(len(object_ids)):
            if r not in matched_obj_indices:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]

        # Build output and register new detections
        tracked_results = []
        for c, d in enumerate(detections):
            if c in matched_box_indices:
                # Find matching object id
                for r in matched_obj_indices:
                    if int(np.argmax(iou_matrix[r])) == c:
                        oid = object_ids[r]
                        item = d.copy()
                        item["track_id"] = oid
                        tracked_results.append(item)
                        break
            else:
                # Register new
                oid = self.next_object_id
                self.next_object_id += 1
                self.objects[oid] = d["box"]
                self.disappeared[oid] = 0
                item = d.copy()
                item["track_id"] = oid
                tracked_results.append(item)
                self._check_counting_line(oid, d["box"], counting_line_y)

        return tracked_results

    def _check_counting_line(self, obj_id: int, box: List[float], line_y: float | None):
        if line_y is None or obj_id in self.counted_ids:
            return
        _, cy = self._box_centroid(box)
        if cy >= line_y:
            self.counted_ids.add(obj_id)
            self.total_bottles_counted += 1
