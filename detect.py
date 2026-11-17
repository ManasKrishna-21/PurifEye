"""Public entry point for AutoML TFLite baseline detection.

Usage:
    uv run python detect.py --source path/to/image.jpg --conf 0.07 --tiled
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from purifeye.tflite_infer import TFLiteDetector
from purifeye.tiling import generate_tile_coordinates, map_tile_detections_to_full_image
from purifeye.merging import class_aware_nms
from purifeye.plot_theme import PURIFEYE_COLORS

DEFAULT_TFLITE_MODEL = (
    Path(__file__).resolve().parent
    / "Plastic-Bottle-Detection-Models"
    / "TensorFlow Lite"
    / "model-export_iod_tflite-pbset2_20200812060539-2020-08-20T11_27_42.436Z_model.tflite"
)


def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw blue design system bounding boxes with dark halo."""
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        conf = d["confidence"]
        # Outer dark halo (242649 in BGR: 0x49, 0x26, 0x24)
        cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0x49, 0x26, 0x24), 3)
        # Inner Sky Blue reticle (65A7D3 in BGR: 0xD3, 0xA7, 0x65)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0xD3, 0xA7, 0x65), 2)

        label = f"bottle {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # Label pill background: Sky Blue
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), (0xD3, 0xA7, 0x65), -1)
        # Text: Dark Navy
        cv2.putText(out, label, (x1 + 3, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0x49, 0x26, 0x24), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="PurifEye AutoML TFLite Detector")
    parser.add_argument("--source", type=str, required=True, help="Image path or directory")
    parser.add_argument("--model", type=str, default=str(DEFAULT_TFLITE_MODEL), help="Path to .tflite model")
    parser.add_argument("--conf", type=float, default=0.07, help="Confidence threshold")
    parser.add_argument("--tiled", action="store_true", help="Enable 3x3 tiled inference")
    parser.add_argument("--threads", type=int, default=4, help="Interpreter threads")
    parser.add_argument("--output", type=str, default="runs/detect_tflite", help="Output directory")
    args = parser.parse_args()

    detector = TFLiteDetector(args.model, num_threads=args.threads)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(args.source)
    images = [src_path] if src_path.is_file() else list(src_path.glob("*.jpg"))

    print(f"Running TFLite detector on {len(images)} images (tiled={args.tiled}, conf={args.conf})...")
    total_time = 0.0

    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        t0 = time.perf_counter()

        if args.tiled:
            tiles = generate_tile_coordinates(w, h, tile_size=640, overlap=0.20)
            raw_dets = []
            for t in tiles:
                tile_crop = img[t[1]:t[3], t[0]:t[2]]
                td = detector.predict(tile_crop, conf_threshold=args.conf)
                raw_dets.extend(map_tile_detections_to_full_image(td, t))
            dets = class_aware_nms(raw_dets, iou_threshold=0.45, conf_threshold=args.conf)
        else:
            dets = detector.predict(img, conf_threshold=args.conf)

        t1 = time.perf_counter()
        dur_ms = (t1 - t0) * 1000
        total_time += dur_ms
        print(f"[{img_p.name}] Found {len(dets)} bottles in {dur_ms:.1f} ms")

        annotated = draw_detections(img, dets)
        cv2.imwrite(str(out_dir / img_p.name), annotated)

    avg_ms = total_time / max(1, len(images))
    print(f"Done! Saved annotated images to {out_dir}. Average latency: {avg_ms:.1f} ms/image.")


if __name__ == "__main__":
    main()
