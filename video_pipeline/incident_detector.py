"""
incident_detector.py
--------------------
Real-time traffic incident detection engine.

Detects 5 types of incidents using YOLO tracking data + density history:

  1. STOPPED VEHICLE   — tracked vehicle barely moves for N frames
                         (possible breakdown or accident)

  2. DENSITY SPIKE     — traffic density jumps sharply in a short window
                         (sudden congestion arrival or merging traffic)

  3. SUDDEN CLEARANCE  — density drops sharply after being high
                         (vehicles fleeing an incident ahead)

  4. CONGESTION ONSET  — density stays above HIGH threshold for many frames
                         (traffic jam confirmed, not just a blip)

  5. ABNORMAL VEHICLE  — a bounding box is much larger than others
                         (overturned vehicle, large debris, obstruction)

Each incident has:
  - type        : machine-readable key
  - severity    : "CRITICAL" | "WARNING" | "INFO"
  - title       : short human-readable label
  - description : detailed explanation
  - evidence    : dict of supporting numbers
  - timestamp   : HH:MM:SS string
  - time_epoch  : float for chart positioning
"""

import time
import math
from collections import deque, defaultdict
from datetime import datetime


# ── Tunable thresholds ────────────────────────────────────────────────────────

STOPPED_FRAMES_NEEDED = 20        # frames vehicle must be stationary
STOPPED_PIXEL_THRESH  = 10        # max centroid movement (pixels)
STOPPED_MIN_AREA      = 600       # ignore tiny bboxes (far vehicles / noise)

SPIKE_WINDOW          = 8         # frames to look back for spike
SPIKE_DELTA           = 0.22      # minimum density rise to trigger

CLEARANCE_WINDOW      = 6
CLEARANCE_DELTA       = -0.25     # density DROP (negative)
CLEARANCE_MIN_PRIOR   = 0.55      # prior density must have been high

CONGESTION_THRESH     = 0.70
CONGESTION_HOLD       = 12        # consecutive frames above threshold

ABNORMAL_RATIO        = 3.2       # bbox area vs median

COOLDOWN = {
    "stopped_vehicle":  25,
    "density_spike":    12,
    "sudden_clearance": 12,
    "congestion_onset": 30,
    "abnormal_vehicle": 20,
}

VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class IncidentDetector:
    """
    Stateful detector — call update() on every YOLO detection frame.

    Parameters
    ----------
    frame_w, frame_h : video frame dimensions (pixels)
    """

    def __init__(self, frame_w: int = 480, frame_h: int = 360):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Per-track centroid history: id → deque of (cx, cy, area, t)
        self._track_hist: dict = defaultdict(
            lambda: deque(maxlen=STOPPED_FRAMES_NEEDED + 5))

        # Density ring buffer for time-series detectors
        self._density_buf: deque = deque(maxlen=80)

        # Congestion state
        self._congestion_frames  = 0
        self._congestion_alerted = False

        # Cooldown: type → last fired wall-clock time
        self._last_fired: dict = {}

        # Full incident log this session
        self.incidents: list = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, density: float, yolo_results) -> list:
        """
        Call every detection frame.

        Parameters
        ----------
        density      : normalised density [0..1]
        yolo_results : raw result list from model.track()

        Returns
        -------
        new_incidents : list of incident dicts fired THIS frame (may be empty)
        """
        self._density_buf.append(density)
        tracks = self._parse_tracks(yolo_results)
        self._update_track_hist(tracks)

        new = []
        new += self._check_stopped(tracks)
        new += self._check_spike()
        new += self._check_clearance()
        new += self._check_congestion(density)
        new += self._check_abnormal(tracks)

        self.incidents.extend(new)
        return new

    def get_recent(self, n: int = 30) -> list:
        """Return the last n incidents."""
        return self.incidents[-n:]

    def total_count(self) -> int:
        return len(self.incidents)

    def reset(self):
        """Clear all state — call between sessions."""
        self._track_hist.clear()
        self._density_buf.clear()
        self._congestion_frames  = 0
        self._congestion_alerted = False
        self._last_fired.clear()
        self.incidents.clear()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_tracks(self, yolo_results) -> list:
        tracks = []
        if not yolo_results:
            return tracks
        for r in yolo_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            cls_list  = r.boxes.cls.cpu().tolist()
            ids_list  = (r.boxes.id.cpu().tolist()
                         if r.boxes.id is not None
                         else [None] * len(cls_list))
            xyxy_list = r.boxes.xyxy.cpu().tolist()
            for cls_id, tid, xyxy in zip(cls_list, ids_list, xyxy_list):
                name = VEHICLE_NAMES.get(int(cls_id))
                if name is None or tid is None:
                    continue
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)
                if area < STOPPED_MIN_AREA:
                    continue
                tracks.append({
                    "id":   int(tid),
                    "cx":   (x1 + x2) / 2,
                    "cy":   (y1 + y2) / 2,
                    "area": area,
                    "name": name,
                })
        return tracks

    def _update_track_hist(self, tracks: list):
        now = time.time()
        seen = set()
        for t in tracks:
            self._track_hist[t["id"]].append((t["cx"], t["cy"], t["area"], now))
            seen.add(t["id"])
        stale = [tid for tid, h in self._track_hist.items()
                 if tid not in seen and h and (now - h[-1][3]) > 3.0]
        for tid in stale:
            del self._track_hist[tid]

    def _cooldown_ok(self, key: str) -> bool:
        now = time.time()
        if (now - self._last_fired.get(key, 0)) >= COOLDOWN.get(key, 15):
            self._last_fired[key] = now
            return True
        return False

    def _make(self, itype, severity, title, description, evidence=None):
        return {
            "type":        itype,
            "severity":    severity,
            "title":       title,
            "description": description,
            "evidence":    evidence or {},
            "timestamp":   datetime.now().strftime("%H:%M:%S"),
            "time_epoch":  time.time(),
        }

    # ── Detector 1: Stopped vehicle ────────────────────────────────────────────

    def _check_stopped(self, tracks: list) -> list:
        found = []
        for t in tracks:
            hist = self._track_hist[t["id"]]
            if len(hist) < STOPPED_FRAMES_NEEDED:
                continue
            recent   = list(hist)[-STOPPED_FRAMES_NEEDED:]
            xs       = [h[0] for h in recent]
            ys       = [h[1] for h in recent]
            movement = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
            if movement <= STOPPED_PIXEL_THRESH:
                if self._cooldown_ok(f"stopped_{t['id']}"):
                    found.append(self._make(
                        "stopped_vehicle", "CRITICAL",
                        "🛑 Stopped Vehicle",
                        f"A {t['name']} (ID {t['id']}) has not moved for "
                        f"{STOPPED_FRAMES_NEEDED}+ frames — possible breakdown or accident.",
                        {"track_id": t["id"], "vehicle": t["name"],
                         "movement_px": round(movement, 1)},
                    ))
        return found

    # ── Detector 2: Density spike ──────────────────────────────────────────────

    def _check_spike(self) -> list:
        if len(self._density_buf) < SPIKE_WINDOW + 1:
            return []
        buf   = list(self._density_buf)
        delta = buf[-1] - buf[-(SPIKE_WINDOW + 1)]
        if delta >= SPIKE_DELTA and self._cooldown_ok("density_spike"):
            return [self._make(
                "density_spike", "WARNING",
                "⚡ Traffic Spike",
                f"Density jumped +{delta:.2f} over {SPIKE_WINDOW} frames "
                f"({buf[-(SPIKE_WINDOW+1)]:.2f} → {buf[-1]:.2f}). "
                f"Sudden congestion or merging traffic detected.",
                {"delta": round(delta,3), "before": round(buf[-(SPIKE_WINDOW+1)],3),
                 "after": round(buf[-1],3)},
            )]
        return []

    # ── Detector 3: Sudden clearance ──────────────────────────────────────────

    def _check_clearance(self) -> list:
        if len(self._density_buf) < CLEARANCE_WINDOW + 1:
            return []
        buf    = list(self._density_buf)
        before = buf[-(CLEARANCE_WINDOW + 1)]
        after  = buf[-1]
        delta  = after - before
        if (before >= CLEARANCE_MIN_PRIOR
                and delta <= CLEARANCE_DELTA
                and self._cooldown_ok("sudden_clearance")):
            return [self._make(
                "sudden_clearance", "WARNING",
                "⚠️ Sudden Clearance",
                f"Density dropped sharply by {abs(delta):.2f} "
                f"({before:.2f} → {after:.2f}). "
                f"Vehicles may be reacting to an incident ahead.",
                {"drop": round(abs(delta),3), "before": round(before,3),
                 "after": round(after,3)},
            )]
        return []

    # ── Detector 4: Congestion onset ──────────────────────────────────────────

    def _check_congestion(self, density: float) -> list:
        if density >= CONGESTION_THRESH:
            self._congestion_frames += 1
        else:
            self._congestion_frames  = max(0, self._congestion_frames - 2)
            self._congestion_alerted = False

        if (self._congestion_frames >= CONGESTION_HOLD
                and not self._congestion_alerted
                and self._cooldown_ok("congestion_onset")):
            self._congestion_alerted = True
            return [self._make(
                "congestion_onset", "WARNING",
                "🚦 Congestion Building",
                f"Density above {CONGESTION_THRESH:.0%} for "
                f"{self._congestion_frames} consecutive frames. "
                f"A traffic jam is forming.",
                {"density": round(density,3),
                 "frames_high": self._congestion_frames},
            )]
        return []

    # ── Detector 5: Abnormal vehicle size ─────────────────────────────────────

    def _check_abnormal(self, tracks: list) -> list:
        if len(tracks) < 3:
            return []
        areas  = sorted([t["area"] for t in tracks])
        median = areas[len(areas) // 2]
        if median < 200:
            return []
        found = []
        for t in tracks:
            if t["area"] > median * ABNORMAL_RATIO:
                if self._cooldown_ok(f"abnormal_{t['id']}"):
                    found.append(self._make(
                        "abnormal_vehicle", "WARNING",
                        "⚠️ Abnormal Vehicle Size",
                        f"Vehicle ID {t['id']} ({t['name']}) is "
                        f"{t['area']/median:.1f}× the median size — "
                        f"possible overturned vehicle, large debris, or obstruction.",
                        {"track_id": t["id"], "vehicle": t["name"],
                         "area_px": round(t["area"]),
                         "median_px": round(median),
                         "size_ratio": round(t["area"]/median, 2)},
                    ))
        return found