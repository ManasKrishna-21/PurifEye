"""Unified evaluation harness for all models: YOLO, AutoML TFLite, and Exports.

Usage:
    uv run python evaluate.py --model yolov8n.pt --split test
    uv run python evaluate.py --benchmark-all --smoke
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from ultralytics import YOLO
from purifeye.tflite_infer import TFLiteDetector
from purifeye.tiling import generate_tile_coordinates, map_tile_detections_to_full_image
from purifeye.merging import class_aware_nms
from purifeye.metrics import evaluate_detections
from purifeye.plot_theme import apply_purifeye_theme, PURIFEYE_COLORS
import matplotlib.pyplot as plt

REPORTS_DIR = Path("reports")
DEFAULT_TFLITE = (
    Path("Plastic-Bottle-Detection-Models")
    / "TensorFlow Lite"
    / "model-export_iod_tflite-pbset2_20200812060539-2020-08-20T11_27_42.436Z_model.tflite"
)


def load_ground_truths(split_labels_dir: Path, split_images_dir: Path) -> List[Dict[str, Any]]:
    """Load ground truth boxes from YOLO label files."""
    gts = []
    lbl_files = sorted(list(split_labels_dir.glob("*.txt")))
    for lf in lbl_files:
        img_p = split_images_dir / f"{lf.stem}.jpg"
        if not img_p.exists():
            continue
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]

        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    x1 = max(0.0, (xc - bw / 2.0) * w)
                    y1 = max(0.0, (yc - bh / 2.0) * h)
                    x2 = min(float(w), (xc + bw / 2.0) * w)
                    y2 = min(float(h), (yc + bh / 2.0) * h)
                    gts.append({
                        "image_id": lf.stem,
                        "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "class_id": cls_id
                    })
    return gts


def evaluate_yolo(
    weights_path: str | Path,
    images_dir: Path,
    tiled: bool = False,
    conf_thresh: float = 0.25,
    max_images: int | None = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run YOLO inference and collect predictions with latency breakdown."""
    model = YOLO(str(weights_path))
    preds = []
    latencies = []

    img_files = sorted(list(images_dir.glob("*.jpg")))
    if max_images:
        img_files = img_files[:max_images]

    # Warmup
    if img_files:
        dummy = cv2.imread(str(img_files[0]))
        if dummy is not None:
            for _ in range(3):
                _ = model(dummy, verbose=False)

    for img_p in img_files:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        if tiled and (w > 640 or h > 640):
            tiles = generate_tile_coordinates(w, h, tile_size=640, overlap=0.20)
            crops = [img[t[1]:t[3], t[0]:t[2]] for t in tiles]
            batch_res = model(crops, conf=conf_thresh, verbose=False)
            raw = []
            for t, r in zip(tiles, batch_res):
                b = r.boxes.xyxy.cpu().numpy()
                c = r.boxes.conf.cpu().numpy()
                for box, conf in zip(b, c):
                    raw.append({"box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])], "confidence": float(conf), "class_id": 0})
            final_dets = class_aware_nms(raw, iou_threshold=0.45, conf_threshold=conf_thresh)
        else:
            res = model(img, imgsz=640 if tiled else 512, conf=conf_thresh, verbose=False)[0]
            b = res.boxes.xyxy.cpu().numpy()
            c = res.boxes.conf.cpu().numpy()
            final_dets = [{"box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])], "confidence": float(conf), "class_id": 0} for box, conf in zip(b, c)]

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        for d in final_dets:
            preds.append({
                "image_id": img_p.stem,
                "box": d["box"],
                "confidence": d["confidence"],
                "class_id": d["class_id"],
            })

    lat_med = float(np.median(latencies)) if latencies else 0.0
    lat_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    return preds, {"median_ms": round(lat_med, 1), "p95_ms": round(lat_p95, 1)}


def evaluate_tflite(
    model_path: str | Path,
    images_dir: Path,
    tiled: bool = False,
    conf_thresh: float = 0.07,
    max_images: int | None = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run TFLite baseline inference."""
    detector = TFLiteDetector(model_path, num_threads=4)
    preds = []
    latencies = []

    img_files = sorted(list(images_dir.glob("*.jpg")))
    if max_images:
        img_files = img_files[:max_images]

    # Warmup
    if img_files:
        dummy = cv2.imread(str(img_files[0]))
        if dummy is not None:
            for _ in range(2):
                _ = detector.predict(dummy)

    for img_p in img_files:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        if tiled and (w > 640 or h > 640):
            tiles = generate_tile_coordinates(w, h, tile_size=640, overlap=0.20)
            raw = []
            for t in tiles:
                crop = img[t[1]:t[3], t[0]:t[2]]
                td = detector.predict(crop, conf_threshold=conf_thresh)
                raw.extend(map_tile_detections_to_full_image(td, t))
            final_dets = class_aware_nms(raw, iou_threshold=0.45, conf_threshold=conf_thresh)
        else:
            final_dets = detector.predict(img, conf_threshold=conf_thresh)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        for d in final_dets:
            preds.append({
                "image_id": img_p.stem,
                "box": d["box"],
                "confidence": d["confidence"],
                "class_id": 0,
            })

    lat_med = float(np.median(latencies)) if latencies else 0.0
    lat_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    return preds, {"median_ms": round(lat_med, 1), "p95_ms": round(lat_p95, 1)}


def run_full_benchmark(yolo_weights: str = "yolov8n.pt", split: str = "test", smoke: bool = False) -> Dict[str, Any]:
    """Execute evaluation across all model variants and produce unified comparison table."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = Path("yolo_dataset/images") / split
    labels_dir = Path("yolo_dataset/labels") / split

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Dataset split {split} not found. Run prepare_yolo_dataset.py first.")
        return {}

    gts = load_ground_truths(labels_dir, images_dir)
    max_imgs = 15 if smoke else None
    tflite_max_imgs = 10 if smoke else 150  # Fixed 150-image subset for slow baseline

    results_table = []

    print(f"Evaluating models on {split} split ({len(gts)} ground truth boxes)...")

    # 1. YOLOv8n Accurate (Tiled)
    print("Evaluating YOLOv8 (Tiled/Accurate)...")
    y_tiled_preds, y_tiled_lat = evaluate_yolo(yolo_weights, images_dir, tiled=True, max_images=max_imgs)
    y_tiled_metrics = evaluate_detections(y_tiled_preds, gts)
    results_table.append({
        "model": "YOLOv8 Nano (Accurate Tiled)",
        "input": "640x640 Tiled",
        "precision": f"{y_tiled_metrics['precision']*100:.1f}%",
        "recall": f"{y_tiled_metrics['recall']*100:.1f}%",
        "map50": f"{y_tiled_metrics['map50']*100:.1f}%",
        "map50_95": f"{y_tiled_metrics['map50_95']*100:.1f}%",
        "latency_cpu": f"~{y_tiled_lat['median_ms']} ms",
        "raw_metrics": y_tiled_metrics,
        "raw_lat": y_tiled_lat
    })

    # 2. YOLOv8n Live (Single-Pass 512)
    print("Evaluating YOLOv8 (Live 512)...")
    y_live_preds, y_live_lat = evaluate_yolo(yolo_weights, images_dir, tiled=False, max_images=max_imgs)
    y_live_metrics = evaluate_detections(y_live_preds, gts)
    results_table.append({
        "model": "YOLOv8 Nano (Live)",
        "input": "512x512",
        "precision": f"{y_live_metrics['precision']*100:.1f}%",
        "recall": f"{y_live_metrics['recall']*100:.1f}%",
        "map50": f"{y_live_metrics['map50']*100:.1f}%",
        "map50_95": f"{y_live_metrics['map50_95']*100:.1f}%",
        "latency_cpu": f"~{y_live_lat['median_ms']} ms",
        "raw_metrics": y_live_metrics,
        "raw_lat": y_live_lat
    })

    # 3. AutoML TFLite Baseline
    if DEFAULT_TFLITE.exists():
        print("Evaluating AutoML TFLite Baseline...")
        tf_preds, tf_lat = evaluate_tflite(DEFAULT_TFLITE, images_dir, tiled=False, max_images=tflite_max_imgs)
        tf_metrics = evaluate_detections(tf_preds, gts)
        results_table.append({
            "model": "AutoML TFLite (baseline)",
            "input": "512x512",
            "precision": f"{tf_metrics['precision']*100:.1f}%",
            "recall": f"{tf_metrics['recall']*100:.1f}%",
            "map50": f"{tf_metrics['map50']*100:.1f}%",
            "map50_95": f"{tf_metrics['map50_95']*100:.1f}%",
            "latency_cpu": f"~{tf_lat['median_ms']} ms",
            "raw_metrics": tf_metrics,
            "raw_lat": tf_lat
        })

    # Print Table
    print("\n" + "="*80)
    print(f"{'Model':<30} | {'Input':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'Latency':<15}")
    print("="*80)
    for r in results_table:
        print(f"{r['model']:<30} | {r['input']:<15} | {r['precision']:<10} | {r['recall']:<10} | {r['map50']:<10} | {r['latency_cpu']:<15}")
    print("="*80 + "\n")

    # Save to reports
    out_json = REPORTS_DIR / "evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"split": split, "smoke": smoke, "results": results_table}, f, indent=2)

    # Plot PR curves
    apply_purifeye_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results_table:
        pr = r["raw_metrics"]["pr_curve"]
        if pr["recall"]:
            ax.plot(pr["recall"], pr["precision"], label=f"{r['model']} (mAP {r['map50']})")
    ax.set_title(f"Precision-Recall Curves ({split.upper()} Split)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(facecolor=PURIFEYE_COLORS["card"], edgecolor=PURIFEYE_COLORS["border"])
    plt.tight_layout()
    plt.savefig(str(REPORTS_DIR / "pr_curve.png"), dpi=150)
    plt.close()

    return {"results": results_table}


def main():
    parser = argparse.ArgumentParser(description="PurifEye Unified Evaluation")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke evaluation")
    args = parser.parse_args()

    run_full_benchmark(yolo_weights=args.weights, split=args.split, smoke=args.smoke)


if __name__ == "__main__":
    main()
