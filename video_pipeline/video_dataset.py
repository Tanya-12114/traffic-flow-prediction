"""
video_dataset.py
----------------
PyTorch Dataset wrapping video clips for the CNN-BiLSTM model.

Accepts precomputed data directly from the app so YOLO only runs
ONCE and vehicle counts are never lost to cache.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List

from video_pipeline.video_extractor import video_to_traffic_series, frames_to_clips


class VideoTrafficDataset(Dataset):
    """
    Attributes
    ----------
    detection_method : 'yolo' | 'pixel' | 'cached'
    vehicle_counts   : list of per-frame dicts {car, truck, bus, motorcycle, total}
    vehicle_summary  : aggregated stats dict (populated when counts available)
    """

    def __init__(
        self,
        video_path:  str,
        clip_len:    int  = 12,
        step:        int  = 1,
        resize:      Tuple[int, int] = (64, 64),
        fps_keep:    int  = 5,
        use_flow:    bool = True,
        use_yolo:    bool = True,
        cache_dir:   Optional[str] = None,
        precomputed: Optional[dict] = None,
    ):
        self.clip_len         = clip_len
        self.resize           = resize
        self.use_flow         = use_flow
        self.detection_method = "unknown"
        self.vehicle_counts   = []
        self.vehicle_summary  = {}

        # ── Get frames + density + counts ──────────────────────
        if precomputed:
            # Best path: use data already computed in the app
            frames                = precomputed["frames"]
            density               = precomputed["density"]
            self.detection_method = precomputed.get("method", "yolo")
            self.vehicle_counts   = precomputed.get("counts", [])
        else:
            frames, density, self.detection_method, self.vehicle_counts = \
                self._load_or_cache(video_path, resize, fps_keep,
                                    use_flow, use_yolo, cache_dir)

        # ── Build vehicle_summary ───────────────────────────────
        if self.vehicle_counts:
            totals = [c["total"] for c in self.vehicle_counts]
            self.vehicle_summary = {
                "total_frames_analysed":      len(totals),
                "max_vehicles_in_frame":      max(totals),
                "min_vehicles_in_frame":      min(totals),
                "avg_vehicles_per_frame":     round(sum(totals) / len(totals), 2),
                "peak_frame_index":           totals.index(max(totals)),
                "total_cars_detected":        sum(c["car"]        for c in self.vehicle_counts),
                "total_trucks_detected":      sum(c["truck"]      for c in self.vehicle_counts),
                "total_buses_detected":       sum(c["bus"]        for c in self.vehicle_counts),
                "total_motorcycles_detected": sum(c["motorcycle"] for c in self.vehicle_counts),
            }
            self.vehicle_summary["grand_total_detections"] = (
                self.vehicle_summary["total_cars_detected"]   +
                self.vehicle_summary["total_trucks_detected"] +
                self.vehicle_summary["total_buses_detected"]  +
                self.vehicle_summary["total_motorcycles_detected"]
            )

        # ── Build (X, y) sliding-window pairs ──────────────────
        clips = frames_to_clips(frames, clip_len=clip_len, step=step)
        N     = clips.shape[0]

        target_indices = np.arange(clip_len, clip_len + N)
        valid          = target_indices < len(density)
        clips          = clips[valid]
        targets        = density[target_indices[valid]]

        clips_t = np.transpose(clips, (0, 1, 4, 2, 3))   # (N,T,H,W,C)→(N,T,C,H,W)
        self.X  = torch.tensor(clips_t, dtype=torch.float32)
        self.y  = torch.tensor(targets,  dtype=torch.float32)

        print(f"[VideoTrafficDataset] X={self.X.shape}  y={self.y.shape}  "
              f"method={self.detection_method}  "
              f"counts={'yes (' + str(len(self.vehicle_counts)) + ' frames)' if self.vehicle_counts else 'no'}")

    def _load_or_cache(
        self, video_path, resize, fps_keep, use_flow, use_yolo, cache_dir
    ) -> Tuple[np.ndarray, np.ndarray, str, list]:

        f_path = None
        d_path = None

        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            tag    = (f"r{resize[0]}x{resize[1]}_fps{fps_keep}"
                      f"_flow{int(use_flow)}_yolo{int(use_yolo)}")
            f_path = cache_path / f"frames_{tag}.npy"
            d_path = cache_path / f"density_{tag}.npy"

            if f_path.exists() and d_path.exists():
                print(f"[VideoTrafficDataset] Loading cache from {cache_dir}")
                frames  = np.load(f_path)
                density = np.load(d_path)
                # Re-run YOLO just for counts (fast, no frame extraction)
                counts = []
                method = "cached"
                if use_yolo:
                    try:
                        from video_pipeline.vehicle_detector import count_vehicles_by_type
                        import cv2
                        cap = cv2.VideoCapture(video_path)
                        idx = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if idx % fps_keep == 0:
                                c = count_vehicles_by_type(frame, model_size="n",
                                                           confidence=0.3)
                                counts.append(c)
                            idx += 1
                        cap.release()
                        method = "yolo"
                        print(f"[VideoTrafficDataset] Counts re-run: {len(counts)} frames")
                    except Exception as e:
                        print(f"[VideoTrafficDataset] Count re-run failed: {e}")
                return frames, density, method, counts

        # No cache hit — process video fresh
        print(f"[VideoTrafficDataset] Processing video: {video_path}")
        frames, density, method, counts = video_to_traffic_series(
            video_path,
            resize   = resize,
            fps_keep = fps_keep,
            use_flow = use_flow,
            use_yolo = use_yolo,
        )

        if f_path and d_path:
            np.save(f_path, frames)
            np.save(d_path, density)
            print(f"[VideoTrafficDataset] Cached to {cache_dir}")

        return frames, density, method, counts

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def input_shape(self):
        return tuple(self.X.shape[1:])