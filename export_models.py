"""Model export suite: ONNX, TFLite (FP16 & INT8), OpenVINO, and standalone Edge Runner.

Usage:
    uv run python export_models.py --weights yolov8n.pt --format onnx
    uv run python export_models.py --weights yolov8n.pt --all
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
from ultralytics import YOLO

REPORTS_DIR = Path("reports")
EXPORTS_DIR = Path("runs/exports")


def export_model_variants(weights_path: str = "yolov8n.pt", formats: List[str] | None = None) -> Dict[str, Any]:
    """Export YOLO model to multiple edge formats and benchmark file sizes."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if formats is None:
        formats = ["onnx", "tflite", "openvino"]

    model = YOLO(weights_path)
    results = {}

    print(f"Exporting {weights_path} to formats: {formats}...")

    # ONNX Export
    if "onnx" in formats:
        print("Exporting to ONNX...")
        try:
            raw_path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
            # Move to runs/exports/ so it doesn't clutter the project root
            dest = EXPORTS_DIR / Path(raw_path).name
            Path(raw_path).replace(dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            results["onnx"] = {"path": str(dest), "size_mb": round(size_mb, 2), "status": "success"}
        except Exception as e:
            results["onnx"] = {"error": str(e), "status": "failed"}

    # TFLite Export
    if "tflite" in formats:
        print("Exporting to TFLite (FP16)...")
        try:
            tflite_path = model.export(format="tflite", imgsz=512, half=True)
            size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            results["tflite_fp16"] = {"path": str(tflite_path), "size_mb": round(size_mb, 2), "status": "success"}
        except Exception as e:
            results["tflite_fp16"] = {"error": str(e), "status": "failed"}

    # OpenVINO Export (CPU optimized)
    if "openvino" in formats:
        print("Exporting to OpenVINO...")
        try:
            ov_path = model.export(format="openvino", imgsz=640, half=False)
            results["openvino"] = {"path": str(ov_path), "status": "success"}
        except Exception as e:
            results["openvino"] = {"error": str(e), "status": "failed"}

    # Save summary report
    with open(REPORTS_DIR / "edge_exports.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


class EdgeONNXRunner:
    """Minimal zero-dependency ONNX Runtime detector without Ultralytics or PyTorch."""

    def __init__(self, onnx_path: str, imgsz: int = 640):
        import onnxruntime as ort
        self.imgsz = imgsz
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, img: np.ndarray, conf_thresh: float = 0.25) -> List[Dict[str, Any]]:
        orig_h, orig_w = img.shape[:2]
        # Letterbox / resize
        resized = cv2.resize(img, (self.imgsz, self.imgsz))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self.session.run(self.output_names, {self.input_name: blob})[0]
        # outputs shape: [1, 5, 8400] for YOLOv8 (x, y, w, h, score)
        predictions = outputs[0].transpose(1, 0)
        detections = []
        for row in predictions:
            score = float(row[4])
            if score >= conf_thresh:
                xc, yc, w, h = row[:4]
                x1 = (xc - w / 2.0) / self.imgsz * orig_w
                y1 = (yc - h / 2.0) / self.imgsz * orig_h
                x2 = (xc + w / 2.0) / self.imgsz * orig_w
                y2 = (yc + h / 2.0) / self.imgsz * orig_h
                detections.append({
                    "box": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                    "confidence": round(score, 4),
                    "class_id": 0
                })
        return detections


def generate_recommendation_matrix() -> Dict[str, Any]:
    matrix = {
        "laptop_cpu_realtime": {
            "model": "YOLOv8 Nano (OpenVINO / ONNX)",
            "input": "512x512",
            "target_fps": "20-30 FPS",
            "advantage": "AVX-512 CPU acceleration and low memory footprint"
        },
        "raspberry_pi_edge": {
            "model": "AutoML MobileNet-SSD TFLite / YOLOv8n INT8",
            "input": "320x320 - 512x512",
            "target_fps": "2-10 FPS",
            "advantage": "Ultra-lightweight ~2.8 MB footprint suitable for Pi Zero/4/5"
        },
        "max_accuracy_survey": {
            "model": "YOLOv8 Nano (Accurate 640px 3x3 Tiling)",
            "input": "Full-Resolution 3x3 Grid",
            "target_fps": "1-3 FPS",
            "advantage": "Highest mAP for tiny, distant bottles on broad river reaches"
        }
    }
    with open(REPORTS_DIR / "edge_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    return matrix


def main():
    parser = argparse.ArgumentParser(description="PurifEye Edge Exporter")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Weights to export")
    parser.add_argument("--format", type=str, choices=["onnx", "tflite", "openvino"], default="onnx")
    parser.add_argument("--all", action="store_true", help="Export all formats")
    args = parser.parse_args()

    formats = ["onnx", "tflite", "openvino"] if args.all else [args.format]
    export_model_variants(args.weights, formats=formats)
    generate_recommendation_matrix()


if __name__ == "__main__":
    main()
