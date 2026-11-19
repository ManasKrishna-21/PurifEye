"""Unified evaluation metrics computation: Precision, Recall, F1, mAP@50, mAP@50-95, and size breakdowns."""

from typing import List, Dict, Any, Tuple
import numpy as np


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between N boxes and M boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-8)
    return inter / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_detections(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_thresholds: List[float] | None = None
) -> Dict[str, Any]:
    """Evaluate detection predictions against ground truths.

    predictions: list of dicts with 'image_id', 'box' [x1, y1, x2, y2], 'confidence', 'class_id'
    ground_truths: list of dicts with 'image_id', 'box' [x1, y1, x2, y2], 'class_id'
    """
    if iou_thresholds is None:
        iou_thresholds = [0.50 + 0.05 * i for i in range(10)]  # 0.50 to 0.95

    # Group ground truths by image_id
    gt_by_img: Dict[str, List[Dict[str, Any]]] = {}
    total_gts = len(ground_truths)
    for gt in ground_truths:
        gt_by_img.setdefault(gt["image_id"], []).append(gt)

    # Sort all predictions by confidence descending
    preds_sorted = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    num_preds = len(preds_sorted)

    if total_gts == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "map50": 0.0, "map50_95": 0.0, "optimal_f1_conf": 0.25,
            "size_breakdown": {}, "pr_curve": {"precision": [], "recall": []}
        }

    # Evaluate at IoU 0.50
    tp_50 = np.zeros(num_preds)
    fp_50 = np.zeros(num_preds)

    matched_gt: Dict[Tuple[str, int], set] = {}

    for i, pred in enumerate(preds_sorted):
        img_id = pred["image_id"]
        gts = gt_by_img.get(img_id, [])
        if not gts:
            fp_50[i] = 1
            continue

        pred_box = np.array(pred["box"])[None, :]
        gt_boxes = np.array([g["box"] for g in gts])
        ious = compute_iou_matrix(pred_box, gt_boxes)[0]

        best_idx = int(np.argmax(ious))
        best_iou = ious[best_idx]

        if best_iou >= 0.50 and (img_id, best_idx) not in matched_gt:
            tp_50[i] = 1
            matched_gt[(img_id, best_idx)] = True
        else:
            fp_50[i] = 1

    cumsum_tp = np.cumsum(tp_50)
    cumsum_fp = np.cumsum(fp_50)

    recalls_50 = cumsum_tp / total_gts
    precisions_50 = cumsum_tp / np.maximum(cumsum_tp + cumsum_fp, 1e-8)

    map50 = compute_ap(recalls_50, precisions_50)

    # Multi-IoU mAP@50-95
    ap_list = [map50]
    for iou_th in iou_thresholds[1:]:
        tp_i = np.zeros(num_preds)
        fp_i = np.zeros(num_preds)
        matched_i = set()
        for idx, pred in enumerate(preds_sorted):
            img_id = pred["image_id"]
            gts = gt_by_img.get(img_id, [])
            if not gts:
                fp_i[idx] = 1
                continue
            pred_box = np.array(pred["box"])[None, :]
            gt_boxes = np.array([g["box"] for g in gts])
            ious = compute_iou_matrix(pred_box, gt_boxes)[0]
            best_idx = int(np.argmax(ious))
            if ious[best_idx] >= iou_th and (img_id, best_idx) not in matched_i:
                tp_i[idx] = 1
                matched_i.add((img_id, best_idx))
            else:
                fp_i[idx] = 1
        c_tp = np.cumsum(tp_i)
        c_fp = np.cumsum(fp_i)
        recs = c_tp / total_gts
        precs = c_tp / np.maximum(c_tp + c_fp, 1e-8)
        ap_list.append(compute_ap(recs, precs))

    map50_95 = float(np.mean(ap_list))

    # Optimal F1 threshold search
    f1_scores = 2 * (precisions_50 * recalls_50) / np.maximum(precisions_50 + recalls_50, 1e-8)
    if len(f1_scores) > 0 and np.max(f1_scores) > 0:
        best_f1_idx = int(np.argmax(f1_scores))
        best_f1 = float(f1_scores[best_f1_idx])
        best_conf = float(preds_sorted[best_f1_idx]["confidence"])
        final_p = float(precisions_50[best_f1_idx])
        final_r = float(recalls_50[best_f1_idx])
    else:
        best_f1, best_conf, final_p, final_r = 0.0, 0.25, 0.0, 0.0

    # Size bucket breakdowns (Small: area < 1024, Med: 1024-9216, Large: > 9216)
    size_breakdown = {}
    for cat_name, (min_a, max_a) in [("small", (0, 1024)), ("medium", (1024, 9216)), ("large", (9216, float("inf")))]:
        cat_gts = [g for g in ground_truths if min_a <= (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1]) < max_a]
        cat_preds = [p for p in predictions if min_a <= (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]) < max_a]
        size_breakdown[cat_name] = {
            "gt_count": len(cat_gts),
            "pred_count": len(cat_preds),
            "recall": round(len(cat_preds) / max(1, len(cat_gts)), 4)
        }

    return {
        "precision": round(final_p, 4),
        "recall": round(final_r, 4),
        "f1": round(best_f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "optimal_f1_conf": round(best_conf, 4),
        "size_breakdown": size_breakdown,
        "pr_curve": {
            "precision": [round(float(p), 4) for p in precisions_50[::max(1, len(precisions_50)//50)]],
            "recall": [round(float(r), 4) for r in recalls_50[::max(1, len(recalls_50)//50)]],
        }
    }
