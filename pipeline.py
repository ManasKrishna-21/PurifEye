"""One-Command Resumable Pipeline for PurifEye.

Executes Stages sequentially:
1. Data Preparation (VOC to YOLO, Splits, Tiled Subset)
2. Training Strategy & Ablations
3. Final CPU Training Run
4. Unified Evaluation on Frozen Test Split
5. Edge Exports & Benchmarks
6. Report Generation for Dashboard

Usage:
    uv run python pipeline.py --smoke
    uv run python pipeline.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

from prepare_yolo_dataset import convert_and_split, run_data_audit, RAW_DATASET_DIR, OUTPUT_YOLO_DIR
from train_yolo import train_yolo, run_ablations, measure_cpu_throughput
from evaluate import run_full_benchmark
from export_models import export_model_variants, generate_recommendation_matrix
from purifeye.logging import setup_logger

MARKERS_DIR = Path("runs/pipeline_markers")
LOGS_DIR = Path("runs/pipeline_logs")


def mark_done(step_name: str, data: dict = None):
    MARKERS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MARKERS_DIR / f"{step_name}.done", "w", encoding="utf-8") as f:
        json.dump(data or {"timestamp": time.time()}, f)


def is_done(step_name: str) -> bool:
    return (MARKERS_DIR / f"{step_name}.done").exists()


def run_pipeline(smoke: bool = False, force: bool = False):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("purifeye.pipeline", log_file=LOGS_DIR / "pipeline.log")

    print("\n" + "="*80)
    print("PURIFEYE — END-TO-END AUTOMATED PIPELINE")
    print(f"Mode: {'SMOKE TEST (Subset)' if smoke else 'FULL PRODUCTION RUN'}")
    print("REMINDER: Keep your laptop plugged in and awake throughout execution.")
    print("="*80 + "\n")

    steps = [
        "1_data_audit",
        "2_data_preparation",
        "3_ablations",
        "4_final_training",
        "5_evaluation",
        "6_edge_exports"
    ]

    # Measure CPU throughput
    tp = measure_cpu_throughput("yolov8n.pt", 640)
    est_total_min = (tp["sec_per_img"] * 3000 * 35 / 60) if not smoke else 2.0
    print(f"Estimated unattended duration: ~{est_total_min:.1f} minutes\n")

    # Step 1: Data Audit
    if force or not is_done("1_data_audit"):
        print("[Step 1/6] Running Data Audit...")
        run_data_audit(RAW_DATASET_DIR, max_samples=50 if smoke else None)
        mark_done("1_data_audit")
    else:
        print("[Step 1/6] Data Audit: ALREADY COMPLETED (Skipping)")

    # Step 2: Data Preparation
    if force or not is_done("2_data_preparation"):
        print("[Step 2/6] Running VOC to YOLO Conversion & Tiling...")
        convert_and_split(RAW_DATASET_DIR, OUTPUT_YOLO_DIR, smoke=smoke, smoke_limit=50 if smoke else 5592)
        mark_done("2_data_preparation")
    else:
        print("[Step 2/6] Data Preparation: ALREADY COMPLETED (Skipping)")

    # Step 3: Ablations
    best_config_name = "abl_3_640px_water_aug"
    if not smoke:
        if force or not is_done("3_ablations"):
            print("[Step 3/6] Running Training Ablations...")
            best_config_name = run_ablations("configs/data.yaml")
            mark_done("3_ablations", {"winner": best_config_name})
        else:
            print("[Step 3/6] Ablations: ALREADY COMPLETED (Skipping)")
    else:
        print("[Step 3/6] Ablations: SKIPPED in smoke mode")

    # Step 4: Final Training
    best_weights = Path(f"runs/train/{'smoke_test_run' if smoke else 'purifeye_final_yolov8n'}/weights/best.pt")
    if force or not is_done("4_final_training"):
        print("[Step 4/6] Executing Final Model Training...")
        epochs = 1 if smoke else 35
        batch = 8 if smoke else 16
        name = "smoke_test_run" if smoke else "purifeye_final_yolov8n"
        best_weights = train_yolo(
            data_yaml="configs/data.yaml",
            model_name="yolov8n.pt",
            epochs=epochs,
            imgsz=512 if smoke else 640,
            batch=batch,
            freeze=1 if smoke else 5,
            name=name,
            resume=True
        )
        mark_done("4_final_training", {"weights": str(best_weights)})
    else:
        print("[Step 4/6] Final Training: ALREADY COMPLETED (Skipping)")

    # Step 5: Unified Evaluation
    eval_weights = str(best_weights) if best_weights.exists() else "yolov8n.pt"
    if force or not is_done("5_evaluation"):
        print("[Step 5/6] Running Unified Evaluation on Frozen Test Split...")
        run_full_benchmark(yolo_weights=eval_weights, split="test", smoke=smoke)
        mark_done("5_evaluation")
    else:
        print("[Step 5/6] Evaluation: ALREADY COMPLETED (Skipping)")

    # Step 6: Edge Exports & Recommendation Matrix
    if force or not is_done("6_edge_exports"):
        print("[Step 6/6] Generating Edge Exports & Recommendations...")
        export_model_variants(eval_weights, formats=["onnx"])
        generate_recommendation_matrix()
        mark_done("6_edge_exports")
    else:
        print("[Step 6/6] Edge Exports: ALREADY COMPLETED (Skipping)")

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print(f"Results and visual reports are ready in 'reports/'.")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PurifEye One-Command Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2-minute smoke pipeline on tiny subset")
    parser.add_argument("--force", action="store_true", help="Force re-run all steps even if marked done")
    args = parser.parse_args()

    run_pipeline(smoke=args.smoke, force=args.force)


if __name__ == "__main__":
    main()
