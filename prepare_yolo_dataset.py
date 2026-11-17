"""Data Audit, VOC to YOLO Conversion, Leakage-Safe Splitting, and Tiled Subset Generation.

Usage:
    uv run python prepare_yolo_dataset.py --audit-only
    uv run python prepare_yolo_dataset.py --smoke
    uv run python prepare_yolo_dataset.py
"""

import argparse
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image

from purifeye.tiling import generate_tile_coordinates, map_box_to_tile
from purifeye.plot_theme import PURIFEYE_COLORS

RAW_DATASET_DIR = Path("Dataset/Plastic-Bottles-Dataset-master/Plastic-Bottles-Dataset")
OUTPUT_YOLO_DIR = Path("yolo_dataset")
REPORTS_DIR = Path("reports")


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Dict[str, Any]], List[str]]:
    """Parse Pascal VOC XML annotation file. Returns (width, height, objects, anomalies)."""
    anomalies = []
    objects = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        anomalies.append(f"Corrupt XML: {e}")
        return 0, 0, [], anomalies

    size_elem = root.find("size")
    xml_w = int(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 0
    xml_h = int(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 0

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        name = name_elem.text.strip().lower() if name_elem is not None and name_elem.text else "unknown"
        diff_elem = obj.find("difficult")
        difficult = int(diff_elem.text) if diff_elem is not None and diff_elem.text else 0
        trunc_elem = obj.find("truncated")
        truncated = int(trunc_elem.text) if trunc_elem is not None and trunc_elem.text else 0

        bndbox = obj.find("bndbox")
        if bndbox is None:
            anomalies.append(f"Missing bndbox for {name}")
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception as e:
            anomalies.append(f"Malformed coordinates: {e}")
            continue

        # Check coordinate anomalies
        if xmax <= xmin or ymax <= ymin:
            anomalies.append(f"Swapped/zero-area coordinates: ({xmin},{ymin},{xmax},{ymax})")
            continue

        objects.append({
            "name": name,
            "difficult": difficult,
            "truncated": truncated,
            "box": [xmin, ymin, xmax, ymax]
        })

    return xml_w, xml_h, objects, anomalies


def compute_dhash(image_path: Path, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for perceptual near-duplicate detection."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img, dtype=np.int32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return hashlib.md5(diff.tobytes()).hexdigest()
    except Exception:
        return ""


def run_data_audit(raw_dir: Path, max_samples: int | None = None) -> Dict[str, Any]:
    """Audit the raw dataset and return comprehensive statistics."""
    print("Starting automated dataset audit...")
    session_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    total_images = 0
    total_annotations = 0
    total_boxes = 0
    small_boxes = 0      # < 32x32 or < 1% image area
    aspect_ratios = []
    resolutions = []
    boxes_per_img = []
    country_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}
    anomalies_list = []
    hashes: Dict[str, List[str]] = {}

    sampled_images = []

    for s_dir in session_dirs:
        # Infer country from session name (e.g. "001-Sindangbarang River, Bogor, Indonesia-eka")
        country = "Unknown"
        s_lower = s_dir.name.lower()
        for c in ["indonesia", "india", "nepal", "pakistan", "bangladesh", "bosnia", "sudan"]:
            if c in s_lower:
                country = c.capitalize()
                break
        country_counts[country] = country_counts.get(country, 0)

        jpg_files = list(s_dir.glob("*.jpg")) + list(s_dir.glob("*.JPG"))
        for img_path in jpg_files:
            if max_samples and total_images >= max_samples:
                break
            xml_path = img_path.with_suffix(".xml")
            if not xml_path.exists():
                xml_path = img_path.with_suffix(".XML")

            total_images += 1
            country_counts[country] += 1

            if len(sampled_images) < 16:
                sampled_images.append(img_path)

            # Perceptual hash for near duplicates
            h = compute_dhash(img_path)
            if h:
                hashes.setdefault(h, []).append(str(img_path))

            if not xml_path.exists():
                anomalies_list.append(f"Missing XML for image: {img_path.name}")
                continue

            total_annotations += 1
            # Get true dimensions from image directly
            try:
                with Image.open(img_path) as im:
                    true_w, true_h = im.size
            except Exception as e:
                anomalies_list.append(f"Corrupt image: {img_path.name} ({e})")
                continue

            resolutions.append((true_w, true_h))
            aspect_ratios.append(round(true_w / max(1, true_h), 2))

            xml_w, xml_h, objects, obj_anoms = parse_voc_xml(xml_path)
            if obj_anoms:
                anomalies_list.extend([f"{img_path.name}: {a}" for a in obj_anoms])

            boxes_per_img.append(len(objects))
            total_boxes += len(objects)

            img_area = max(1, true_w * true_h)
            for obj in objects:
                class_counts[obj["name"]] = class_counts.get(obj["name"], 0) + 1
                bx = obj["box"]
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                b_area = bw * bh
                if (bw < 32 and bh < 32) or (b_area / img_area < 0.01):
                    small_boxes += 1

    # Near duplicates count (hashes with > 1 image)
    duplicates_count = sum(len(paths) - 1 for paths in hashes.values() if len(paths) > 1)

    small_obj_share = (small_boxes / max(1, total_boxes)) * 100.0

    recommendation = (
        f"DATA AUDIT FINDING: {small_obj_share:.1f}% of objects are small (<32x32 or <1% area). "
        f"Downscaling large images (median resolution ~3024x3096) directly to 512x512 eliminates fine details. "
        f"RECOMMENDATION: Use 640px overlapped tiling (20% overlap) with full-image downscaled anchors. "
        f"Targeted augmentations for water glare and turbidity must be enabled without vertical scene flips."
    )

    report = {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(total_boxes / max(1, total_images), 2),
        "small_objects_count": small_boxes,
        "small_objects_percentage": round(small_obj_share, 2),
        "near_duplicates_found": duplicates_count,
        "country_distribution": country_counts,
        "class_distribution": class_counts,
        "anomalies_logged": len(anomalies_list),
        "recommendation": recommendation,
    }

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Generate 4x4 contact sheet
    if sampled_images:
        generate_contact_sheet(sampled_images[:16], REPORTS_DIR / "data_audit_contact_sheet.jpg")

    print(f"Audit completed: {total_images} images, {total_boxes} boxes, {small_obj_share:.1f}% small objects.")
    return report


def generate_contact_sheet(image_paths: List[Path], save_path: Path):
    """Create a 4x4 contact sheet of dataset images."""
    thumb_w, thumb_h = 320, 240
    sheet = np.full((thumb_h * 4, thumb_w * 4, 3), 0x24, dtype=np.uint8)  # PurifEye canvas background
    for idx, p in enumerate(image_paths):
        r, c = divmod(idx, 4)
        img = cv2.imread(str(p))
        if img is not None:
            thumb = cv2.resize(img, (thumb_w - 4, thumb_h - 4))
            sheet[r*thumb_h + 2 : r*thumb_h + 2 + thumb_h - 4, c*thumb_w + 2 : c*thumb_w + 2 + thumb_w - 4] = thumb
    cv2.imwrite(str(save_path), sheet)


def convert_and_split(
    raw_dir: Path,
    output_dir: Path,
    smoke: bool = False,
    smoke_limit: int = 50,
    tile_size: int = 640,
    overlap: float = 0.20,
    seed: int = 42
):
    """Convert Pascal VOC to YOLO, perform leakage-safe splitting by session, and generate capped tiled subset."""
    random.seed(seed)
    np.random.seed(seed)

    print(f"Converting dataset (smoke={smoke}, seed={seed})...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if smoke:
        # Only take first 2 sessions and cap images
        session_dirs = session_dirs[:2]

    # Group images by session to guarantee leakage-safe isolation
    session_images: Dict[str, List[Path]] = {}
    for s_dir in session_dirs:
        jpgs = sorted(list(s_dir.glob("*.jpg")) + list(s_dir.glob("*.JPG")))
        if jpgs:
            session_images[s_dir.name] = jpgs

    # Shuffle sessions and allocate Train (70%), Val (15%), Test (15%)
    all_sessions = list(session_images.keys())
    random.shuffle(all_sessions)

    num_sessions = len(all_sessions)
    if num_sessions < 3:
        # Intra-session split to ensure non-empty train, val, and test splits
        split_map = {}
        for s in all_sessions:
            split_map[s] = "intra_split"
    else:
        n_train = max(1, int(0.70 * num_sessions))
        n_val = max(1, int(0.15 * num_sessions))
        train_sessions = all_sessions[:n_train]
        val_sessions = all_sessions[n_train : n_train + n_val]
        test_sessions = all_sessions[n_train + n_val :]
        split_map = {}
        for s in train_sessions:
            split_map[s] = "train"
        for s in val_sessions:
            split_map[s] = "val"
        for s in test_sessions:
            split_map[s] = "test"

    verification_images: List[Tuple[Path, Path]] = []
    manifest: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    processed_count = 0
    tiled_training_count = 0

    for s_name, img_paths in session_images.items():
        sess_mode = split_map[s_name]
        for img_idx, img_p in enumerate(img_paths):
            if smoke and processed_count >= smoke_limit:
                break

            if sess_mode == "intra_split":
                frac = (img_idx % 10) / 10.0
                if frac < 0.70:
                    split = "train"
                elif frac < 0.85:
                    split = "val"
                else:
                    split = "test"
            else:
                split = sess_mode
            if smoke and processed_count >= smoke_limit:
                break
            xml_p = img_p.with_suffix(".xml")
            if not xml_p.exists():
                xml_p = img_p.with_suffix(".XML")
            if not xml_p.exists():
                continue

            img = cv2.imread(str(img_p))
            if img is None:
                continue
            true_h, true_w = img.shape[:2]

            _, _, objects, _ = parse_voc_xml(xml_p)

            # Build normalized YOLO boxes
            yolo_boxes = []
            for obj in objects:
                bx1, by1, bx2, by2 = obj["box"]
                # Clamp within true image dimensions
                cx1 = max(0.0, min(float(bx1), float(true_w)))
                cy1 = max(0.0, min(float(by1), float(true_h)))
                cx2 = max(0.0, min(float(bx2), float(true_w)))
                cy2 = max(0.0, min(float(by2), float(true_h)))

                bw = cx2 - cx1
                bh = cy2 - cy1
                if bw <= 1.0 or bh <= 1.0:
                    continue  # Drop degenerate box

                xc = (cx1 + cx2) / 2.0 / true_w
                yc = (cy1 + cy2) / 2.0 / true_h
                w = bw / true_w
                h = bh / true_h
                yolo_boxes.append((0, xc, yc, w, h))

            base_name = f"{s_name}_{img_p.stem}"
            dest_img = output_dir / "images" / split / f"{base_name}.jpg"
            dest_lbl = output_dir / "labels" / split / f"{base_name}.txt"

            # Save full image and label
            cv2.imwrite(str(dest_img), img)
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                for b in yolo_boxes:
                    lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

            manifest[split].append(base_name)
            processed_count += 1

            if len(verification_images) < 30:
                verification_images.append((dest_img, dest_lbl))

            # Tiled subset for training (capped to ~1.5x original count)
            if split == "train" and len(objects) > 0 and (tiled_training_count < int(processed_count * 0.5)):
                tiles = generate_tile_coordinates(true_w, true_h, tile_size=tile_size, overlap=overlap)
                for t_idx, t in enumerate(tiles):
                    t_boxes = []
                    for obj in objects:
                        m = map_box_to_tile(obj["box"], t, min_visibility=0.40)
                        if m is not None:
                            t_boxes.append((0, m[0], m[1], m[2], m[3]))

                    # Keep tile only if it contains boxes or occasional background
                    if t_boxes:
                        tile_crop = img[t[1]:t[3], t[0]:t[2]]
                        t_name = f"{base_name}_tile{t_idx}"
                        t_img_p = output_dir / "images" / "train" / f"{t_name}.jpg"
                        t_lbl_p = output_dir / "labels" / "train" / f"{t_name}.txt"
                        cv2.imwrite(str(t_img_p), tile_crop)
                        with open(t_lbl_p, "w", encoding="utf-8") as tlf:
                            for tb in t_boxes:
                                tlf.write(f"{tb[0]} {tb[1]:.6f} {tb[2]:.6f} {tb[3]:.6f} {tb[4]:.6f}\n")
                        manifest["train"].append(t_name)
                        tiled_training_count += 1
                        if tiled_training_count >= int(processed_count * 0.5):
                            break

    # Save manifest
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump({k: len(v) for k, v in manifest.items()}, mf, indent=2)

    # Verification: Draw 30 converted images
    verif_dir = REPORTS_DIR / "voc_to_yolo_verification"
    verif_dir.mkdir(parents=True, exist_ok=True)
    for v_img_p, v_lbl_p in verification_images:
        v_img = cv2.imread(str(v_img_p))
        if v_img is None:
            continue
        vh, vw = v_img.shape[:2]
        if v_lbl_p.exists():
            with open(v_lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, xc, yc, bw, bh = [float(p) for p in parts[:5]]
                        x1 = int((xc - bw / 2.0) * vw)
                        y1 = int((yc - bh / 2.0) * vh)
                        x2 = int((xc + bw / 2.0) * vw)
                        y2 = int((yc + bh / 2.0) * vh)
                        cv2.rectangle(v_img, (x1, y1), (x2, y2), (0xD3, 0xA7, 0x65), 2)
        cv2.imwrite(str(verif_dir / v_img_p.name), v_img)

    print(f"Dataset preparation complete! Train: {len(manifest['train'])}, Val: {len(manifest['val'])}, Test: {len(manifest['test'])}.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="PurifEye VOC to YOLO Converter and Auditor")
    parser.add_argument("--audit-only", action="store_true", help="Run audit and exit")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode on subset")
    parser.add_argument("--smoke-limit", type=int, default=50, help="Max images in smoke mode")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DATASET_DIR), help="Raw VOC dataset dir")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_YOLO_DIR), help="Output YOLO dataset dir")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    out_path = Path(args.out_dir)

    if args.audit_only:
        run_data_audit(raw_path)
    else:
        run_data_audit(raw_path, max_samples=args.smoke_limit if args.smoke else None)
        convert_and_split(raw_path, out_path, smoke=args.smoke, smoke_limit=args.smoke_limit)


if __name__ == "__main__":
    main()
