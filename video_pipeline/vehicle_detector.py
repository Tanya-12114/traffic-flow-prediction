"""
vehicle_detector.py
-------------------
YOLO11-based vehicle detector with tracking.

Two modes:
  1. count_vehicles_by_type()   — per-frame counts (used for density labels)
  2. count_unique_vehicles()    — tracks vehicles across frames using ByteTrack,
                                  returns UNIQUE vehicle count (each vehicle = 1)

COCO class IDs:  2=car | 3=motorcycle | 5=bus | 7=truck
Install:         pip install ultralytics
"""

import cv2
import subprocess
import sys

VEHICLE_IDS   = {2, 3, 5, 7}
VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MAX_VEHICLES  = 20
_MODEL_CACHE  = {}


def _get_model(model_size: str = "n"):
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]
    try:
        from ultralytics import YOLO
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "ultralytics", "--quiet"])
        from ultralytics import YOLO
    model = YOLO(f"yolo11{model_size}.pt")
    _MODEL_CACHE[model_size] = model
    return model


# ── Per-frame counting (for density labels) ───────────────────────────────────

def count_vehicles_by_type(
    frame_bgr,
    model_size: str   = "n",
    confidence: float = 0.3,
) -> dict:
    """
    Count vehicles in ONE frame (no tracking).
    Same car in next frame = counted again.
    Used only for per-frame density labels fed to the BiLSTM.

    Returns: {car, truck, bus, motorcycle, total}
    """
    model   = _get_model(model_size)
    results = model(frame_bgr, conf=confidence, verbose=False)
    counts  = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
    for r in results:
        for cls_id in r.boxes.cls.cpu().tolist():
            name = VEHICLE_NAMES.get(int(cls_id))
            if name:
                counts[name] += 1
    counts["total"] = sum(counts.values())
    return counts


def count_vehicles_yolo(frame_bgr, model_size="n", confidence=0.3) -> int:
    return count_vehicles_by_type(frame_bgr, model_size, confidence)["total"]


def frame_density_yolo(frame_bgr, model_size="n", confidence=0.3,
                       max_vehicles=MAX_VEHICLES) -> float:
    return min(count_vehicles_yolo(frame_bgr, model_size, confidence) / max_vehicles, 1.0)


def annotate_frame(frame_bgr, model_size="n", confidence=0.3):
    model   = _get_model(model_size)
    results = model(frame_bgr, conf=confidence, verbose=False)
    return results[0].plot()


# ── Unique vehicle tracking across entire video ───────────────────────────────

def count_unique_vehicles(
    video_path:  str,
    model_size:  str   = "n",
    confidence:  float = 0.3,
    fps_keep:    int   = 1,       # kept for API compat — ignored (auto-computed)
    target_fps:  float = 8.0,     # process at this effective FPS for speed
    max_width:   int   = 640,     # resize frames before inference
) -> dict:
    """
    Track vehicles across the video using ByteTrack and return unique counts.

    Speed optimisations vs the original:
      1. Frame-skip  — auto-calculates a skip factor so we process at
                       `target_fps` instead of every raw frame.
                       e.g. 30 fps video → process 1 in every 7-8 frames.
      2. Resize      — downscales frames to max_width before YOLO inference
                       (YOLO11n is already small; halving resolution ~4× faster).
      3. imgsz hint  — passes the resized width directly to YOLO so it skips
                       its own internal resize step.
      4. No-op skip  — frames that are skipped are read with cap.grab()
                       (no decode) rather than cap.read() (decode + copy).

    Parameters
    ----------
    video_path  : path to video file
    model_size  : 'n' | 's' | 'm'
    confidence  : detection threshold
    target_fps  : desired processing rate (frames per second of video time)
    max_width   : maximum frame width sent to YOLO

    Returns
    -------
    {
        "unique_total":       int,
        "unique_cars":        int,
        "unique_trucks":      int,
        "unique_buses":       int,
        "unique_motorcycles": int,
        "frames_processed":   int,
    }
    """
    model = _get_model(model_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    # ── Auto frame-skip from video FPS ──────────────────────────────────────
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip      = max(1, int(round(src_fps / target_fps)))   # e.g. 30fps→skip=7

    # ── Resize factor ────────────────────────────────────────────────────────
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or max_width
    scale     = min(1.0, max_width / src_w)
    imgsz     = int((src_w * scale) // 32 * 32) or 320   # must be multiple of 32

    # ── Tracking state ───────────────────────────────────────────────────────
    seen_ids = {"car": set(), "truck": set(), "bus": set(), "motorcycle": set()}
    frames_processed = 0
    frame_idx        = 0

    while True:
        if frame_idx % skip != 0:
            # Fast seek — decode skipped, no memory alloc
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # Downscale if needed
        if scale < 1.0:
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        results = model.track(
            frame,
            conf      = confidence,
            persist   = True,
            verbose   = False,
            tracker   = "bytetrack.yaml",
            imgsz     = imgsz,
        )

        for r in results:
            if r.boxes.id is None:
                continue
            for cls_id, track_id in zip(
                r.boxes.cls.cpu().tolist(),
                r.boxes.id.cpu().tolist()
            ):
                name = VEHICLE_NAMES.get(int(cls_id))
                if name:
                    seen_ids[name].add(int(track_id))

        frames_processed += 1
        frame_idx        += 1

    cap.release()

    return {
        "unique_cars":        len(seen_ids["car"]),
        "unique_trucks":      len(seen_ids["truck"]),
        "unique_buses":       len(seen_ids["bus"]),
        "unique_motorcycles": len(seen_ids["motorcycle"]),
        "unique_total":       sum(len(s) for s in seen_ids.values()),
        "frames_processed":   frames_processed,
    }