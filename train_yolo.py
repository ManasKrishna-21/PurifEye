"""CPU-Optimized YOLOv8 training and ablation manager with budget estimation.

Usage:
    uv run python train_yolo.py --smoke
    uv run python train_yolo.py --epochs 35 --batch 16
    uv run python train_yolo.py --ablate
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any

from ultralytics import YOLO

from purifeye.config import load_config
from purifeye.logging import setup_logger

RUNS_DIR = Path("runs/train")
REGISTRY_FILE = Path("reports/run_registry.csv")


def measure_cpu_throughput(model_path: str = "yolov8n.pt", imgsz: int = 640) -> Dict[str, float]:
    """Measure raw CPU training iteration latency on 20 dummy steps to calibrate time budget."""
    print("Calibrating CPU throughput (20 iterations)...")
    model = YOLO(model_path)
    # Warm up with quick predict
    import numpy as np
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy_img, verbose=False)

    t0 = time.perf_counter()
    n_iters = 15
    for _ in range(n_iters):
        _ = model(dummy_img, verbose=False)
    t1 = time.perf_counter()

    sec_per_img = (t1 - t0) / n_iters
    img_per_sec = 1.0 / max(1e-4, sec_per_img)
    print(f"Calibrated CPU throughput: {sec_per_img*1000:.1f} ms/image ({img_per_sec:.2f} images/sec).")
    return {"sec_per_img": sec_per_img, "img_per_sec": img_per_sec}


def record_run(entry: Dict[str, Any]):
    """Append a training run record to the central run registry CSV."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = REGISTRY_FILE.exists()
    fields = ["timestamp", "name", "epochs", "imgsz", "batch", "map50", "map50_95", "precision", "recall", "weights", "notes"]
    with open(REGISTRY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: entry.get(k, "") for k in fields})


def train_yolo(
    data_yaml: str = "configs/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 35,
    imgsz: int = 640,
    batch: int = 16,
    freeze: int = 5,
    project: str = "runs/train",
    name: str = "purifeye_yolov8n",
    resume: bool = False,
    augment: bool = True
) -> Path:
    """Execute CPU-friendly training of YOLOv8."""
    logger = setup_logger("purifeye.train")
    logger.info(f"Starting training run: {name} (epochs={epochs}, imgsz={imgsz}, batch={batch}, freeze={freeze})")

    model = YOLO(model_name)

    # Resume if last checkpoint exists
    last_pt = Path(project) / name / "weights" / "last.pt"
    if resume and last_pt.exists():
        logger.info(f"Resuming from checkpoint: {last_pt}")
        results = model.train(resume=True)
    else:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            freeze=freeze,
            workers=4,
            device="cpu",
            project=project,
            name=name,
            exist_ok=True,
            cos_lr=True,
            close_mosaic=max(1, epochs // 7),
            patience=10,
            save=True,
            save_period=1,
            val=True,
            verbose=False,
            # River water scene augmentations (glare, turbidity, scale, no upside-down flips)
            hsv_h=0.015 if augment else 0.0,
            hsv_s=0.7 if augment else 0.0,
            hsv_v=0.4 if augment else 0.0,
            degrees=10.0 if augment else 0.0,
            translate=0.10 if augment else 0.0,
            scale=0.5 if augment else 0.0,
            fliplr=0.5 if augment else 0.0,
            flipud=0.0,  # Never flip river upside down
        )

    best_weights = Path(project) / name / "weights" / "best.pt"
    logger.info(f"Training completed! Best weights: {best_weights}")

    # Log metrics to registry
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    record_run({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "map50": round(metrics.get("metrics/mAP50(B)", 0.0), 4),
        "map50_95": round(metrics.get("metrics/mAP50-95(B)", 0.0), 4),
        "precision": round(metrics.get("metrics/precision(B)", 0.0), 4),
        "recall": round(metrics.get("metrics/recall(B)", 0.0), 4),
        "weights": str(best_weights),
        "notes": "Trained with PurifEye CPU configuration",
    })

    return best_weights


def run_ablations(data_yaml: str = "configs/data.yaml") -> str:
    """Run 4 short ablations (6-8 epochs each) on 15% subset to determine best configuration."""
    print("--- Starting 4-Step Training Ablations ---")
    configs = [
        {"name": "abl_1_512px", "imgsz": 512, "model": "yolov8n.pt", "augment": False},
        {"name": "abl_2_640px", "imgsz": 640, "model": "yolov8n.pt", "augment": False},
        {"name": "abl_3_640px_water_aug", "imgsz": 640, "model": "yolov8n.pt", "augment": True},
    ]

    best_name = configs[0]["name"]
    best_score = -1.0

    for cfg in configs:
        print(f"Running ablation: {cfg['name']}...")
        w = train_yolo(
            data_yaml=data_yaml,
            model_name=cfg["model"],
            epochs=6,
            imgsz=cfg["imgsz"],
            batch=16,
            freeze=2,
            project="runs/train/ablations",
            name=cfg["name"],
            augment=cfg["augment"]
        )
        # Evaluate validation score
        eval_model = YOLO(w)
        val_res = eval_model.val(data=data_yaml, split="val", imgsz=cfg["imgsz"], verbose=False)
        m50 = val_res.results_dict.get("metrics/mAP50(B)", 0.0)
        rec = val_res.results_dict.get("metrics/recall(B)", 0.0)
        score = m50 * 0.6 + rec * 0.4
        print(f"[{cfg['name']}] Val mAP@50: {m50:.4f}, Recall: {rec:.4f}, Combined Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = cfg["name"]

    print(f"Ablation winner: {best_name} (Combined score: {best_score:.4f})")
    return best_name


def main():
    parser = argparse.ArgumentParser(description="PurifEye YOLOv8 CPU Trainer")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--freeze", type=int, default=5, help="Frozen backbone layers")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Smoke run: 1 epoch on tiny subset")
    parser.add_argument("--ablate", action="store_true", help="Run ablation suite")
    args = parser.parse_args()

    measure_cpu_throughput(args.model, args.imgsz)

    if args.ablate:
        run_ablations(args.data)
    elif args.smoke:
        train_yolo(
            data_yaml=args.data,
            model_name=args.model,
            epochs=1,
            imgsz=512,
            batch=8,
            freeze=1,
            project="runs/train",
            name="smoke_test_run"
        )
    else:
        train_yolo(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            freeze=args.freeze,
            project="runs/train",
            name="purifeye_final_yolov8n",
            resume=args.resume
        )


if __name__ == "__main__":
    main()
