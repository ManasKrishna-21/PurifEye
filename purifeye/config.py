"""Configuration loader and management for PurifEye."""

from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return get_default_config()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_default_config() -> Dict[str, Any]:
    """Fallback default configuration."""
    return {
        "hardware": {
            "threads": 4,
            "workers": 4,
            "batch_size": 16,
            "cache_ram": False,
        },
        "dataset": {
            "raw_dir": "Dataset/Plastic-Bottles-Dataset-master/Plastic-Bottles-Dataset",
            "yolo_dir": "yolo_dataset",
            "tile_size": 640,
            "tile_overlap": 0.20,
            "min_visibility": 0.40,
            "split_ratio": [0.70, 0.15, 0.15],
            "seed": 42,
        },
        "training": {
            "model": "yolov8n.pt",
            "imgsz": 640,
            "epochs": 40,
            "patience": 10,
            "freeze": 5,
            "cos_lr": True,
            "close_mosaic": 5,
        },
        "inference": {
            "accurate": {
                "tiling": True,
                "tile_size": 640,
                "overlap": 0.20,
                "conf_thresh": 0.25,
                "iou_thresh": 0.45,
            },
            "live": {
                "tiling": False,
                "imgsz": 512,
                "conf_thresh": 0.30,
                "iou_thresh": 0.50,
            },
        },
    }
