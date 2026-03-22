"""
live_camera.py
--------------
Real-time traffic prediction from a live camera feed.

Supports:
  - Webcam / USB camera  (index 0, 1, 2 ...)
  - RTSP stream URL      (rtsp://...)
  - HTTP stream URL      (http://...)
  - Video file (loop)    (path to .mp4 / .avi)

New features:
  C — Online adaptive learning: RealTimePredictor optionally wraps an
      OnlineLearner that continuously fine-tunes the model on recent frames.
  D — Incident alerts: density spike detection with configurable thresholds.
      Alerts fire when density crosses HIGH threshold or rises sharply.
"""

import cv2
import numpy as np
import torch
import time
from collections import deque
from datetime import datetime
from typing import Optional


VEHICLE_IDS   = {2, 3, 5, 7}
VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ── Feature D: alert thresholds ──────────────────────────────────────────────
ALERT_HIGH_DENSITY   = 0.66   # density above this → HIGH traffic alert
ALERT_SPIKE_DELTA    = 0.20   # density rise over 5 frames → SPIKE alert
ALERT_COOLDOWN_SECS  = 15     # seconds between repeated alerts of same type


class LiveCameraStream:
    """
    Wraps a camera/stream source.

    Parameters
    ----------
    source  : int (webcam index), str (RTSP/HTTP URL), or str (video file path)
    resize  : (W, H) for display and processing
    loop    : if True and source is a video file, restart from frame 0 when
              the file ends — simulates a continuous live feed for testing
    """

    def __init__(self, source=0, resize=(640, 480), loop=False):
        self.source = source
        self.resize = resize
        self.loop   = loop
        self.cap    = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self.source}\n"
                f"For webcam: use index 0 or 1.\n"
                f"For RTSP: use rtsp://user:pass@ip:port/stream\n"
                f"For video file: provide a valid .mp4 / .avi path"
            )
        return self

    def read(self):
        """Returns (success, frame_bgr). Loops video file if loop=True."""
        if self.cap is None:
            raise RuntimeError("Call open() first.")
        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if ret and self.resize:
            frame = cv2.resize(frame, self.resize)
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    @property
    def fps(self):
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS) or 25
        return 25


class RealTimePredictor:
    """
    Runs YOLO + optional BiLSTM on a live camera stream.

    Parameters
    ----------
    model           : trained CNN_BiLSTM_Attention (or None for YOLO-only)
    yolo_model_size : 'n' | 's' | 'm'
    confidence      : YOLO confidence threshold
    max_vehicles    : vehicle count → density 1.0
    clip_len        : BiLSTM sequence length (must match training)
    kalman_smoother : KalmanSmoother instance
    process_every   : run YOLO every N frames
    use_online_learning : Feature C — enable incremental model fine-tuning
    online_update_every : fine-tune after every N new detections
    """

    def __init__(
        self,
        model                = None,
        yolo_model_size      = "n",
        confidence           = 0.3,
        max_vehicles         = 20,
        clip_len             = 12,
        kalman_smoother      = None,
        process_every        = 3,
        use_online_learning  = False,   # Feature C
        online_update_every  = 30,      # Feature C
    ):
        self.model              = model
        self.confidence         = confidence
        self.max_vehicles       = max_vehicles
        self.clip_len           = clip_len
        self.smoother           = kalman_smoother
        self.process_every      = process_every
        self._frame_count       = 0
        self._yolo              = None
        self._model_size        = yolo_model_size

        # Rolling history
        self.density_history  = deque(maxlen=200)
        self.count_history    = deque(maxlen=200)
        self.pred_history     = deque(maxlen=200)
        self.smoothed_history = deque(maxlen=200)
        self.timestamps       = deque(maxlen=200)

        # Unique / passed-by vehicle tracking
        self.unique_ids  = {"car": set(), "truck": set(), "bus": set(), "motorcycle": set()}
        self._active_ids = {"car": set(), "truck": set(), "bus": set(), "motorcycle": set()}
        self.passed_ids  = {"car": set(), "truck": set(), "bus": set(), "motorcycle": set()}

        # Latest state
        self.current_counts  = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "total": 0}
        self.current_density = 0.0
        self.current_pred    = 0.0
        self.traffic_level   = "Unknown"



        # ── Feature C: Online learner ────────────────────────────────────────
        self.online_learner = None
        if use_online_learning and model is not None:
            try:
                from training.online_learner import OnlineLearner
                self.online_learner = OnlineLearner(
                    model        = model,
                    buffer_size  = 200,
                    update_every = online_update_every,
                    lr_finetune  = 1e-5,
                )
                print(f"[OnlineLearner] Enabled — update every {online_update_every} samples")
            except Exception as e:
                print(f"[OnlineLearner] Could not initialise: {e}")

        # ── Feature D: Alert state ───────────────────────────────────────────
        self.alerts            = []           # list of alert dicts
        self._last_alert_time  = {}           # alert_type → last fired timestamp
        self._alert_active     = False        # is a HIGH alert currently active?

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_yolo(self):
        if self._yolo is None:
            from video_pipeline.vehicle_detector import _get_model
            self._yolo = _get_model(self._model_size)
        return self._yolo

    def _detect(self, frame_bgr):
        """
        Run YOLO+ByteTrack. Updates unique_ids, passed_ids, _active_ids.
        Returns (counts_dict, results).
        """
        yolo  = self._load_yolo()
        h, w  = frame_bgr.shape[:2]
        imgsz = int(w // 32 * 32) or 320
        results = yolo.track(frame_bgr, conf=self.confidence,
                             persist=True, verbose=False,
                             tracker="bytetrack.yaml", imgsz=imgsz)

        counts     = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        new_active = {"car": set(), "truck": set(), "bus": set(), "motorcycle": set()}

        for r in results:
            boxes    = r.boxes
            cls_list = boxes.cls.cpu().tolist()
            ids_list = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(cls_list)
            for cls_id, track_id in zip(cls_list, ids_list):
                name = VEHICLE_NAMES.get(int(cls_id))
                if name:
                    counts[name] += 1
                    if track_id is not None:
                        tid = int(track_id)
                        self.unique_ids[name].add(tid)
                        new_active[name].add(tid)

        for name in self._active_ids:
            gone = self._active_ids[name] - new_active[name]
            self.passed_ids[name].update(gone)

        self._active_ids = new_active
        counts["total"]  = sum(counts.values())
        return counts, results

    def _bilstm_predict(self, frame_bgr):
        """Run the BiLSTM on the last clip_len density values."""
        if self.model is None or len(self.density_history) < self.clip_len:
            return self.current_density
        seq    = list(self.density_history)[-self.clip_len:]
        frames = np.array(seq, dtype=np.float32).reshape(self.clip_len, 1, 1, 1)
        frames = np.repeat(np.repeat(frames, 64, axis=2), 64, axis=3)
        clip   = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        model_to_use = (self.online_learner.model
                        if self.online_learner else self.model)
        model_to_use.eval()
        with torch.no_grad():
            pred = model_to_use(clip).item()
        return float(np.clip(pred, 0, 1))

    def _check_alerts(self, density: float) -> list:
        """
        Feature D: check if an incident alert should fire.
        Returns list of new alert dicts fired this frame.
        """
        now        = time.time()
        new_alerts = []

        # Alert type 1: HIGH density threshold crossed
        alert_type = "high_density"
        cooldown_ok = (now - self._last_alert_time.get(alert_type, 0)) > ALERT_COOLDOWN_SECS
        if density >= ALERT_HIGH_DENSITY and cooldown_ok:
            alert = {
                "type":      "high_density",
                "severity":  "HIGH" if density >= 0.85 else "MODERATE",
                "message":   f"High traffic density detected: {density:.2f}",
                "density":   density,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "icon":      "🔴" if density >= 0.85 else "🟡",
            }
            new_alerts.append(alert)
            self._last_alert_time[alert_type] = now
            self._alert_active = True
        elif density < ALERT_HIGH_DENSITY * 0.8:
            self._alert_active = False   # clear alert when density drops

        # Alert type 2: rapid density spike
        alert_type = "spike"
        cooldown_ok = (now - self._last_alert_time.get(alert_type, 0)) > ALERT_COOLDOWN_SECS
        if len(self.density_history) >= 5 and cooldown_ok:
            recent   = list(self.density_history)[-5:]
            delta    = density - recent[0]
            if delta >= ALERT_SPIKE_DELTA:
                alert = {
                    "type":      "spike",
                    "severity":  "SPIKE",
                    "message":   f"Traffic spike: +{delta:.2f} over last 5 readings",
                    "density":   density,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "icon":      "⚡",
                }
                new_alerts.append(alert)
                self._last_alert_time[alert_type] = now

        return new_alerts

    def _annotate(self, frame_bgr, counts, density, pred, level,
                  track_results=None, alerts=None):
        """Draw bounding boxes + HUD. Red border when alert is active."""
        if track_results is not None:
            frame = track_results[0].plot()
        else:
            yolo    = self._load_yolo()
            results = yolo(frame_bgr, conf=self.confidence, verbose=False)
            frame   = results[0].plot()

        h, w = frame.shape[:2]

        # ── Feature D: red alert border ──────────────────────────────────────
        if alerts or self._alert_active:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 220), 4)

        hud_h   = 110 if not self.model else 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        level_colors = {"Low": (80, 200, 80), "Moderate": (0, 165, 255), "High": (60, 60, 220)}
        color = level_colors.get(level, (255, 255, 255))

        cv2.putText(frame, f"Traffic: {level}   Density: {density:.2f}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cur = counts
        cv2.putText(frame,
                    f"Now:  Cars {cur['car']}  Trucks {cur['truck']}  "
                    f"Buses {cur['bus']}  Motos {cur['motorcycle']}  "
                    f"(Total {cur['total']})",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        pb       = self.passed_ids
        pb_total = sum(len(v) for v in pb.values())
        cv2.putText(frame,
                    f"Passed: Cars {len(pb['car'])}  Trucks {len(pb['truck'])}  "
                    f"Buses {len(pb['bus'])}  Motos {len(pb['motorcycle'])}  "
                    f"(Total {pb_total})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 230, 255), 1)
        if self.model:
            cv2.putText(frame, f"BiLSTM pred: {pred:.3f}",
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1)

        # ── Feature D: alert text on frame ───────────────────────────────────
        if alerts:
            for i, alert in enumerate(alerts[-2:]):   # show max 2 on frame
                cv2.putText(frame,
                            f"ALERT: {alert['message']}",
                            (10, h - 30 - i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return frame

    def _get_traffic_level(self, density):
        if   density < 0.33: return "Low"
        elif density < 0.66: return "Moderate"
        else:                return "High"

    # ── Main entry point ─────────────────────────────────────────────────────

    def process_frame(self, frame_bgr):
        """
        Process one frame. Returns (annotated_frame, state_dict).
        """
        self._frame_count += 1
        should_detect = (self._frame_count % self.process_every == 0)

        track_results  = None
        new_alerts     = []

        if should_detect:
            counts, track_results = self._detect(frame_bgr)
            density               = min(counts["total"] / self.max_vehicles, 1.0)

            pred = self._bilstm_predict(frame_bgr)

            level = self._get_traffic_level(density)

            self.current_counts  = counts
            self.current_density = density
            self.current_pred    = pred
            self.traffic_level   = level

            self.density_history.append(density)
            self.count_history.append(counts["total"])
            self.pred_history.append(pred)
            self.timestamps.append(time.time())

            if self.smoother and len(self.density_history) >= 3:
                smoothed = self.smoother.smooth(list(self.density_history))
                self.smoothed_history.append(float(smoothed[-1]))
            else:
                self.smoothed_history.append(density)

            # ── Feature D: check for alerts ───────────────────────────────
            new_alerts = self._check_alerts(density)
            self.alerts.extend(new_alerts)

            # ── Feature C: feed sample to online learner ──────────────────
            # We use the current density as the pseudo-label (ground truth
            # is unavailable in real-time, so we use the YOLO-derived density
            # as a noisy but immediate signal for continual adaptation).
            if self.online_learner is not None and len(self.density_history) >= self.clip_len:
                try:
                    seq    = list(self.density_history)[-self.clip_len:]
                    frames = np.array(seq, dtype=np.float32).reshape(self.clip_len, 1, 1, 1)
                    frames = np.repeat(np.repeat(frames, 64, axis=2), 64, axis=3)
                    X_clip = torch.tensor(frames, dtype=torch.float32)
                    self.online_learner.add_sample(X_clip, density)
                except Exception as e:
                    pass   # never crash the stream due to learner error

        annotated = self._annotate(
            frame_bgr,
            self.current_counts,
            self.current_density,
            self.current_pred,
            self.traffic_level,
            track_results = track_results,
            alerts        = new_alerts,
        )

        # ── Feature C: get learner stats for display ──────────────────────
        learner_stats = (self.online_learner.get_stats()
                         if self.online_learner else None)

        state = {
            "counts":          self.current_counts.copy(),
            "density":         self.current_density,
            "prediction":      self.current_pred,
            "traffic_level":   self.traffic_level,
            "density_history": list(self.density_history),
            "count_history":   list(self.count_history),
            "pred_history":    list(self.pred_history),
            "smoothed_history":list(self.smoothed_history),
            "unique_counts": {
                "car":        len(self.unique_ids["car"]),
                "truck":      len(self.unique_ids["truck"]),
                "bus":        len(self.unique_ids["bus"]),
                "motorcycle": len(self.unique_ids["motorcycle"]),
                "total":      sum(len(v) for v in self.unique_ids.values()),
            },
            "passed_counts": {
                "car":        len(self.passed_ids["car"]),
                "truck":      len(self.passed_ids["truck"]),
                "bus":        len(self.passed_ids["bus"]),
                "motorcycle": len(self.passed_ids["motorcycle"]),
                "total":      sum(len(v) for v in self.passed_ids.values()),
            },
            # Feature C
            "online_learner_stats": learner_stats,

            # Feature D
            "new_alerts":    new_alerts,
            "all_alerts":    list(self.alerts[-20:]),  # last 20 alerts
            "alert_active":  self._alert_active,
        }
        return annotated, state


def encode_frame_jpeg(frame_bgr, quality=80) -> bytes:
    """Encode a BGR frame to JPEG bytes for Streamlit display."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()