"""
video_extractor.py
------------------
Extracts frames + computes vehicle counts via YOLO11.

video_to_traffic_series() returns 4 values:
    frames, density, method, counts_per_frame
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

# ── Import vehicle detector at module level (not lazily inside function) ──
# This ensures import errors are caught early and clearly, not silently swallowed.
try:
    from video_pipeline.vehicle_detector import count_vehicles_by_type, annotate_frame
    YOLO_AVAILABLE = True
except Exception as _yolo_import_err:
    YOLO_AVAILABLE = False
    _YOLO_IMPORT_ERROR = str(_yolo_import_err)

DEFAULT_RESIZE   = (64, 64)
DEFAULT_FPS_KEEP = 5


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path, resize=DEFAULT_RESIZE, fps_keep=DEFAULT_FPS_KEEP,
                   max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % fps_keep == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (resize[1], resize[0]))
            frames.append(gray.astype(np.float32) / 255.0)
            if max_frames and len(frames) >= max_frames: break
        idx += 1
    cap.release()
    if not frames: raise ValueError("No frames extracted.")
    return np.stack(frames, axis=0)[..., np.newaxis]


def extract_optical_flow(video_path, resize=DEFAULT_RESIZE, fps_keep=DEFAULT_FPS_KEEP,
                         max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    result, prev_gray, idx = [], None, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % fps_keep == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (resize[1], resize[0]))
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                           0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag = np.clip(mag, 0, np.percentile(mag, 99) + 1e-6)
                mag = mag / (np.max(mag) + 1e-6)
            else:
                mag = np.zeros_like(gray, dtype=np.float32)
            result.append(np.stack([gray.astype(np.float32)/255.0, mag], axis=-1))
            prev_gray = gray
            if max_frames and len(result) >= max_frames: break
        idx += 1
    cap.release()
    if not result: raise ValueError("No frames extracted.")
    return np.stack(result, axis=0)


def frames_to_clips(frames, clip_len=12, step=1):
    T = frames.shape[0]
    clips = [frames[s: s+clip_len] for s in range(0, T-clip_len, step)]
    return np.stack(clips, axis=0)


# ── Density via YOLO ──────────────────────────────────────────────────────────

def _density_via_yolo(video_path, fps_keep, confidence, max_vehicles, model_size):
    """
    Run YOLO on every sampled frame.
    Returns (density_array, counts_per_frame_list).
    Raises exception if YOLO is not available — caller handles fallback.
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError(f"vehicle_detector import failed: {_YOLO_IMPORT_ERROR}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    counts_list, density_list, idx = [], [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % fps_keep == 0:
            c = count_vehicles_by_type(frame, model_size=model_size,
                                       confidence=confidence)
            counts_list.append(c)
            density_list.append(min(c["total"] / max_vehicles, 1.0))
        idx += 1
    cap.release()

    if not counts_list:
        raise ValueError("No frames processed.")
    return np.array(density_list, dtype=np.float32), counts_list


def _density_via_pixel(frames):
    return np.array([
        float(np.mean(frames[t, ..., 0] > 0.35))
        for t in range(frames.shape[0])
    ], dtype=np.float32)


# ── Main entry point ──────────────────────────────────────────────────────────

def video_to_traffic_series(
    video_path,
    resize             = DEFAULT_RESIZE,
    fps_keep           = DEFAULT_FPS_KEEP,
    use_flow           = True,
    use_yolo           = True,
    yolo_confidence    = 0.3,
    yolo_max_vehicles  = 20,
    yolo_model_size    = "n",
):
    """
    Returns: frames, density, method, counts_per_frame
      method           : 'yolo' | 'pixel'
      counts_per_frame : list of {car, truck, bus, motorcycle, total} per frame
                         EMPTY LIST when method='pixel'
    """
    # Step 1: extract frames for CNN
    if use_flow:
        frames = extract_optical_flow(video_path, resize=resize, fps_keep=fps_keep)
    else:
        frames = extract_frames(video_path, resize=resize, fps_keep=fps_keep)

    # Step 2: density + counts
    method           = "pixel"
    counts_per_frame = []
    yolo_error       = None

    if use_yolo:
        try:
            density, counts_per_frame = _density_via_yolo(
                video_path, fps_keep,
                confidence   = yolo_confidence,
                max_vehicles = yolo_max_vehicles,
                model_size   = yolo_model_size,
            )
            method = "yolo"
        except Exception as e:
            yolo_error = str(e)
            density    = _density_via_pixel(frames)
    else:
        density = _density_via_pixel(frames)

    min_len          = min(len(frames), len(density))
    counts_per_frame = counts_per_frame[:min_len] if counts_per_frame else []

    # Attach yolo_error so the app can display it to the user
    result = (frames[:min_len], density[:min_len], method, counts_per_frame)

    # Store error as attribute on a wrapper so app can surface it
    class Result(tuple):
        pass
    r            = Result(result)
    r.yolo_error = yolo_error
    return r