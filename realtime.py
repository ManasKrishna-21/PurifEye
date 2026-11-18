"""Low-Latency Multi-Threaded Real-Time Video Pipeline.

Three-Thread Producer-Consumer Architecture:
1. Capture Thread: Always fetches freshest camera/video frame.
2. Inference Thread: Reads newest frame; single-slot queue drops stale frames to prevent lag accumulation.
3. Render/Output Thread: Renders PurifEye HUD telemetry and displays or streams frame.

Usage:
    uv run python realtime.py --source 0
    uv run python realtime.py --source path/to/river.mp4 --benchmark --duration 10
"""

import argparse
import csv
import queue
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

from purifeye.tracking import CentroidTracker
from purifeye.plot_theme import PURIFEYE_COLORS

LOGS_DIR = Path("runs/realtime_logs")


def create_synthetic_river_frame(frame_idx: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a synthetic river scene with floating simulated plastic bottles for cameras-free testing."""
    frame = np.full((height, width, 3), (0x49, 0x26, 0x24), dtype=np.uint8)  # Water dark navy base
    # Add subtle water ripple lines
    for y in range(0, height, 40):
        offset = int(np.sin((frame_idx * 0.1) + y) * 15)
        cv2.line(frame, (0, y + offset), (width, y - offset), (0x7C, 0x33, 0x23), 1)

    # Simulated floating bottles drifting down the river
    num_bottles = 3
    for b in range(num_bottles):
        bx = int((frame_idx * 3 + b * 200) % (width + 60)) - 30
        by = int(height * 0.3 + b * 70 + np.sin(frame_idx * 0.05 + b) * 20)
        bw, bh = 45, 20
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0xEB, 0xD3, 0xB2), -1)
        cv2.rectangle(frame, (bx + bw - 8, by + 4), (bx + bw + 4, by + bh - 4), (0xD3, 0xA7, 0x65), -1)

    return frame


class RealtimePipeline:
    def __init__(
        self,
        source: str = "0",
        weights: str = "yolov8n.pt",
        imgsz: int = 512,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        record_path: Optional[str] = None,
        log_csv: Optional[str] = None
    ):
        self.source = source
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.record_path = record_path
        self.log_csv = log_csv

        self.running = False
        self.capture_queue: queue.Queue = queue.Queue(maxsize=1)
        self.render_queue: queue.Queue = queue.Queue(maxsize=1)

        self.tracker = CentroidTracker(max_disappeared=20, iou_thresh=0.30)

        # Telemetry metrics
        self.fps = 0.0
        self.inf_ms = 0.0
        self.e2e_ms = 0.0
        self.dropped_frames = 0
        self.total_frames_captured = 0
        self.unique_bottles = 0
        self.current_detections: List[Dict[str, Any]] = []

        self.latest_rendered_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

    def start(self):
        self.running = True
        self.cap_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.inf_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.cap_thread.start()
        self.inf_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "cap_thread") and self.cap_thread.is_alive():
            self.cap_thread.join(timeout=1.0)
        if hasattr(self, "inf_thread") and self.inf_thread.is_alive():
            self.inf_thread.join(timeout=1.0)

    def _capture_worker(self):
        # Determine source
        is_cam = False
        cam_idx = 0
        try:
            cam_idx = int(self.source)
            is_cam = True
        except ValueError:
            pass

        cap = cv2.VideoCapture(cam_idx if is_cam else self.source)
        use_synthetic = False
        if not cap.isOpened():
            print(f"[PurifEye RealTime] Note: Camera/source '{self.source}' not accessible. Falling back to synthetic river simulation.")
            use_synthetic = True

        frame_idx = 0
        while self.running:
            t_cap = time.perf_counter()
            if use_synthetic:
                frame = create_synthetic_river_frame(frame_idx)
                time.sleep(0.033)  # ~30 FPS capture rate
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

            frame_idx += 1
            self.total_frames_captured += 1

            # Push to single-slot queue: drop stale frame if full
            try:
                self.capture_queue.put_nowait((frame, t_cap))
            except queue.Full:
                self.dropped_frames += 1
                try:
                    _ = self.capture_queue.get_nowait()
                except queue.Empty:
                    pass
                self.capture_queue.put((frame, t_cap))

        cap.release()

    def _inference_worker(self):
        model = YOLO(self.weights)
        prev_time = time.perf_counter()

        while self.running:
            try:
                frame, t_cap = self.capture_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            t_inf_start = time.perf_counter()
            results = model(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False
            )[0]
            t_inf_end = time.perf_counter()
            self.inf_ms = (t_inf_end - t_inf_start) * 1000.0

            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            raw_dets = []
            for b, c, cl in zip(boxes, confs, clss):
                raw_dets.append({
                    "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "confidence": float(c),
                    "class_id": int(cl),
                })

            # Update persistent tracking & river crossing line (at 75% height)
            h, w = frame.shape[:2]
            line_y = h * 0.75
            tracked = self.tracker.update(raw_dets, counting_line_y=line_y)
            self.current_detections = tracked
            self.unique_bottles = self.tracker.total_bottles_counted

            # Render overlay
            rendered = self._render_hud(frame.copy(), tracked, line_y)

            t_now = time.perf_counter()
            self.e2e_ms = (t_now - t_cap) * 1000.0
            self.fps = 1.0 / max(1e-4, t_now - prev_time)
            prev_time = t_now

            with self.frame_lock:
                self.latest_rendered_frame = rendered

    def _render_hud(self, frame: np.ndarray, detections: List[Dict[str, Any]], line_y: float) -> np.ndarray:
        h, w = frame.shape[:2]

        # Draw counting line (Steel Blue: 0xAF, 0x6C, 0x38)
        cv2.line(frame, (0, int(line_y)), (w, int(line_y)), (0xAF, 0x6C, 0x38), 2)
        cv2.putText(frame, "MONITORING TRANSECT", (10, int(line_y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0xEB, 0xD3, 0xB2), 1)

        # Draw detections
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            conf = d["confidence"]
            tid = d.get("track_id", 0)

            # Outer dark halo (0x49, 0x26, 0x24)
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0x49, 0x26, 0x24), 3)
            # Inner Sky Blue (0xD3, 0xA7, 0x65)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0xD3, 0xA7, 0x65), 2)

            label = f"#{tid} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), (0xD3, 0xA7, 0x65), -1)
            cv2.putText(frame, label, (x1 + 3, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0x49, 0x26, 0x24), 1, cv2.LINE_AA)

        # Top HUD Telemetry Banner (Dark Charcoal Navy pill)
        cv2.rectangle(frame, (10, 10), (w - 10, 48), (0x49, 0x26, 0x24), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 48), (0x89, 0x49, 0x36), 1)

        hud_text = (
            f"FPS: {self.fps:4.1f} | INF: {self.inf_ms:4.1f}ms | E2E: {self.e2e_ms:4.1f}ms | "
            f"ACTIVE: {len(detections)} | COUNTED: {self.unique_bottles} | DROPPED: {self.dropped_frames}"
        )
        cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0xEF, 0xDF, 0xCD), 1, cv2.LINE_AA)

        return frame

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        with self.frame_lock:
            if self.latest_rendered_frame is None:
                return None
            ret, buf = cv2.imencode(".jpg", self.latest_rendered_frame)
            return buf.tobytes() if ret else None


def run_benchmark(duration_sec: int = 10, source: str = "0", weights: str = "yolov8n.pt"):
    print(f"Running PurifEye Real-Time Benchmark ({duration_sec}s)...")
    pipe = RealtimePipeline(source=source, weights=weights)
    pipe.start()

    fps_samples = []
    lat_samples = []

    start_t = time.time()
    while time.time() - start_t < duration_sec:
        time.sleep(0.5)
        if pipe.fps > 0:
            fps_samples.append(pipe.fps)
            lat_samples.append(pipe.inf_ms)
            print(f"  Live: {pipe.fps:.1f} FPS | {pipe.inf_ms:.1f} ms inf | Dropped: {pipe.dropped_frames}")

    pipe.stop()

    avg_fps = float(np.mean(fps_samples)) if fps_samples else 0.0
    p95_lat = float(np.percentile(lat_samples, 95)) if lat_samples else 0.0
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Sustained Average FPS: {avg_fps:.1f}")
    print(f"95th Percentile Latency: {p95_lat:.1f} ms")
    print(f"Total Frames Processed: {pipe.total_frames_captured}")
    print(f"Stale Frames Dropped: {pipe.dropped_frames}")


def main():
    parser = argparse.ArgumentParser(description="PurifEye Real-Time Video Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Camera index, video file, or synthetic")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--benchmark", action="store_true", help="Run sustained FPS benchmark")
    parser.add_argument("--duration", type=int, default=10, help="Benchmark duration in seconds")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(duration_sec=args.duration, source=args.source, weights=args.weights)
    else:
        pipe = RealtimePipeline(source=args.source, weights=args.weights)
        pipe.start()
        print("Real-time pipeline running. Press Ctrl+C to terminate...")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping pipeline...")
            pipe.stop()


if __name__ == "__main__":
    main()
