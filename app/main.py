"""FastAPI Backend Server for PurifEye Dashboard.

Serves synchronous endpoints, static assets, MJPEG live stream, and API routes.
"""

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from realtime import RealtimePipeline
from detect_yolo import run_yolo_detection, draw_blue_detections
from detect import draw_detections as draw_tflite_detections, DEFAULT_TFLITE_MODEL
from purifeye.tflite_infer import TFLiteDetector
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
UPLOADS_DIR = BASE_DIR / "runs" / "app_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PurifEye Dashboard", version="0.1.0")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global real-time pipeline instance
pipeline_instance: Optional[RealtimePipeline] = None
yolo_cached_model: Optional[YOLO] = None
tflite_cached_detector: Optional[TFLiteDetector] = None


def get_pipeline() -> RealtimePipeline:
    global pipeline_instance
    if pipeline_instance is None:
        best_pt = BASE_DIR / "runs" / "train" / "purifeye_final_yolov8n" / "weights" / "best.pt"
        weights = str(best_pt) if best_pt.exists() else "yolov8n.pt"
        pipeline_instance = RealtimePipeline(source="0", weights=weights)
        pipeline_instance.start()
    return pipeline_instance


def get_yolo_model() -> YOLO:
    global yolo_cached_model
    if yolo_cached_model is None:
        best_pt = BASE_DIR / "runs" / "train" / "purifeye_final_yolov8n" / "weights" / "best.pt"
        weights = str(best_pt) if best_pt.exists() else "yolov8n.pt"
        yolo_cached_model = YOLO(weights)
    return yolo_cached_model


def get_tflite_detector() -> Optional[TFLiteDetector]:
    global tflite_cached_detector
    if tflite_cached_detector is None and DEFAULT_TFLITE_MODEL.exists():
        tflite_cached_detector = TFLiteDetector(DEFAULT_TFLITE_MODEL, num_threads=4)
    return tflite_cached_detector


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Load evaluation report
    eval_json = REPORTS_DIR / "evaluation_results.json"
    eval_rows = []
    is_smoke = True
    if eval_json.exists():
        try:
            with open(eval_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                eval_rows = data.get("results", [])
                is_smoke = data.get("smoke", True)
        except Exception:
            pass

    # Fallback rows if not evaluated yet
    if not eval_rows:
        eval_rows = [
            {"model": "YOLOv8 Nano (Accurate Tiled)", "input": "640x640 Tiled", "precision": "Pending", "recall": "Pending", "map50": "Pending", "map50_95": "Pending", "latency_cpu": "Pending"},
            {"model": "YOLOv8 Nano (Live)", "input": "512x512", "precision": "Pending", "recall": "Pending", "map50": "Pending", "map50_95": "Pending", "latency_cpu": "Pending"},
            {"model": "AutoML TFLite (baseline)", "input": "512x512", "precision": "55.2%", "recall": "48.7%", "map50": "38.0%", "map50_95": "19.5%", "latency_cpu": "~1,770 ms"},
        ]

    # Load data audit report
    audit_json = REPORTS_DIR / "data_audit_report.json"
    audit_data = {}
    if audit_json.exists():
        try:
            with open(audit_json, "r", encoding="utf-8") as f:
                audit_data = json.load(f)
        except Exception:
            pass

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "eval_rows": eval_rows,
            "audit_data": audit_data,
            "is_smoke": is_smoke,
        }
    )


def gen_video_frames():
    pipe = get_pipeline()
    while True:
        jpeg_bytes = pipe.get_latest_frame_jpeg()
        if jpeg_bytes is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
        time.sleep(0.033)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/telemetry")
def get_telemetry():
    # Only return live data if pipeline already started; don't auto-start on poll.
    global pipeline_instance
    if pipeline_instance is None:
        return JSONResponse({
            "fps": 0.0, "inf_ms": 0.0, "e2e_ms": 0.0,
            "active_detections": 0, "counted_bottles": 0, "dropped_frames": 0,
        })
    pipe = pipeline_instance
    return JSONResponse({
        "fps": round(pipe.fps, 1),
        "inf_ms": round(pipe.inf_ms, 1),
        "e2e_ms": round(pipe.e2e_ms, 1),
        "active_detections": len(pipe.current_detections),
        "counted_bottles": pipe.unique_bottles,
        "dropped_frames": pipe.dropped_frames,
    })


@app.post("/api/analyze")
def analyze_image(
    file: UploadFile = File(...),
    mode: str = Form("accurate"),
    conf: float = Form(0.25)
):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    model = get_yolo_model()
    dets, latency_ms = run_yolo_detection(model, img, mode=mode, conf_thresh=conf)
    annotated = draw_blue_detections(img, dets)

    save_name = f"analyzed_{int(time.time()*1000)}.jpg"
    save_path = UPLOADS_DIR / save_name
    cv2.imwrite(str(save_path), annotated)

    return JSONResponse({
        "bottle_count": len(dets),
        "latency_ms": round(latency_ms, 1),
        "image_url": f"/runs/app_uploads/{save_name}"
    })


@app.post("/api/compare")
def compare_models(file: UploadFile = File(...)):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # 1. YOLO
    yolo_model = get_yolo_model()
    y_dets, y_lat = run_yolo_detection(yolo_model, img, mode="accurate")
    y_annotated = draw_blue_detections(img, y_dets)
    y_name = f"comp_yolo_{int(time.time()*1000)}.jpg"
    cv2.imwrite(str(UPLOADS_DIR / y_name), y_annotated)

    # 2. TFLite Baseline
    tflite_det = get_tflite_detector()
    if tflite_det:
        t0 = time.perf_counter()
        tf_dets = tflite_det.predict(img, conf_threshold=0.07)
        tf_lat = (time.perf_counter() - t0) * 1000.0
        tf_annotated = draw_tflite_detections(img, tf_dets)
        tf_name = f"comp_tflite_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(UPLOADS_DIR / tf_name), tf_annotated)
    else:
        tf_dets, tf_lat = [], 0.0
        tf_name = y_name

    return JSONResponse({
        "yolo": {
            "count": len(y_dets),
            "latency_ms": round(y_lat, 1),
            "image_url": f"/runs/app_uploads/{y_name}"
        },
        "tflite": {
            "count": len(tf_dets),
            "latency_ms": round(tf_lat, 1),
            "image_url": f"/runs/app_uploads/{tf_name}"
        }
    })


# Mount app uploads directory for static serving
app.mount("/runs/app_uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="app_uploads")


def main():
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
