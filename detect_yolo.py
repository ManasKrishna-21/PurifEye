"""Public entry point for YOLOv8 plastic bottle detection.

Usage:
    uv run python detect_yolo.py --source path/to/image.jpg --weights yolov8n.pt --mode accurate
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from purifeye.tiling import generate_tile_coordinates, map_tile_detections_to_full_image
from purifeye.merging import class_aware_nms
from purifeye.plot_theme import PURIFEYE_COLORS


def draw_blue_detections(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes with the Sky Blue (#65A7D3) reticle and dark (#242649) halo."""
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        conf = d["confidence"]
        # Outer dark halo (0x49, 0x26, 0x24)
        cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0x49, 0x26, 0x24), 3)
        # Inner Sky Blue (0xD3, 0xA7, 0x65)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0xD3, 0xA7, 0x65), 2)

        track_str = f" #{d['track_id']}" if "track_id" in d else ""
        label = f"bottle{track_str} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), (0xD3, 0xA7, 0x65), -1)
        cv2.putText(out, label, (x1 + 3, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0x49, 0x26, 0x24), 1, cv2.LINE_AA)
    return out


def run_yolo_detection(
    model: YOLO,
    image: np.ndarray,
    mode: str = "accurate",
    tile_size: int = 640,
    overlap: float = 0.20,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45
) -> tuple[list, float]:
    """Run YOLO detection under specified operating mode (accurate vs live vs adaptive)."""
    h, w = image.shape[:2]
    t0 = time.perf_counter()

    should_tile = False
    if mode == "accurate":
        should_tile = (w > tile_size or h > tile_size)
    elif mode == "adaptive":
        # First check full image; if bottles are very small or image is high-res, tile
        if w > 1280 or h > 1280:
            should_tile = True

    if should_tile:
        tiles = generate_tile_coordinates(w, h, tile_size=tile_size, overlap=overlap)
        tile_crops = [image[t[1]:t[3], t[0]:t[2]] for t in tiles]
        # Run batch prediction on CPU
        batch_results = model(tile_crops, conf=conf_thresh, iou=iou_thresh, verbose=False)
        raw_dets = []
        for t, res in zip(tiles, batch_results):
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            t_dets = []
            for b, c, cl in zip(boxes, confs, clss):
                t_dets.append({"box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "confidence": float(c), "class_id": int(cl)})
            raw_dets.extend(map_tile_detections_to_full_image(t_dets, t))
        dets = class_aware_nms(raw_dets, iou_threshold=iou_thresh, conf_threshold=conf_thresh)
    else:
        results = model(image, imgsz=tile_size if mode == "accurate" else 512, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        dets = []
        for b, c, cl in zip(boxes, confs, clss):
            dets.append({"box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "confidence": float(c), "class_id": int(cl)})

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    return dets, latency_ms


def main():
    parser = argparse.ArgumentParser(description="PurifEye YOLOv8 Detector")
    parser.add_argument("--source", type=str, required=True, help="Image path or directory")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--mode", type=str, choices=["accurate", "live", "adaptive"], default="accurate")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="runs/detect_yolo", help="Output directory")
    args = parser.parse_args()

    model = YOLO(args.weights)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(args.source)
    images = [src_path] if src_path.is_file() else list(src_path.glob("*.jpg"))

    print(f"Running YOLOv8 ({args.mode} mode) on {len(images)} images...")
    total_ms = 0.0

    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        dets, latency_ms = run_yolo_detection(model, img, mode=args.mode, conf_thresh=args.conf)
        total_ms += latency_ms
        print(f"[{img_p.name}] Found {len(dets)} bottles in {latency_ms:.1f} ms")

        annotated = draw_blue_detections(img, dets)
        cv2.imwrite(str(out_dir / img_p.name), annotated)

    avg_ms = total_ms / max(1, len(images))
    print(f"Done! Saved annotated images to {out_dir}. Average latency: {avg_ms:.1f} ms/image.")


if __name__ == "__main__":
    main()
