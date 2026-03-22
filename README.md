# Traffic Flow Prediction System

Real-time traffic density prediction from video using CNN-BiLSTM, YOLO11, and GNN fusion.

## Architecture

```
Video → YOLO11 detection → CNN-BiLSTM prediction → Kalman smoothing → Display
           ↓                        ↑
      ByteTrack IDs          Context encoder
      (unique counts)       (weather + time)
```

### Models
- **CNN-BiLSTM with Temporal Attention** (`models/cnn_bilstm.py`) — shared-weight CNN encodes each frame spatially; BiLSTM models the temporal sequence; attention weights which frames matter most
- **Context Encoder** (`models/context_encoder.py`) — encodes time-of-day, day-of-week, weather, temperature and visibility into a 32-dim vector fused into predictions
- **Kalman Smoother** (`models/kalman.py`) — RTS smoother for noise-free density curves
- **Multi-camera GNN** (`models/graph_fusion.py`) — propagates traffic states across a road network graph built from GPS coordinates

### Pipeline
- **YOLO11 + ByteTrack** (`video_pipeline/vehicle_detector.py`) — detects and tracks vehicles; counts each unique vehicle exactly once
- **Online Learner** (`training/online_learner.py`) — replay-buffer based incremental fine-tuning during live streams
- **Grad-CAM** (`utils/gradcam.py`) — heatmaps showing which road regions drove each prediction

## Quick Start

### UI (recommended)
```bash
pip install -r requirements.txt
streamlit run frontend/app_video.py
```

### CLI
```bash
python main.py --video data/videos/traffic.mp4
python main.py --video data/videos/traffic.mp4 --weather rain --temp 12 --visibility 3
python main.py --help
```

## UI Tabs

| Tab | What it does |
|-----|-------------|
| YouTube | Download a traffic video from YouTube and run full pipeline |
| Upload Video | Upload a local video file and run full pipeline |
| GNN Fusion | Simulate multi-camera road network density fusion |
| Live Camera | Real-time stream from webcam, RTSP, or looped video file |

## Live Stream Features
- Per-frame vehicle type counts (cars, trucks, buses, motorcycles)
- Passed-by tracking — each vehicle counted once as it exits the scene
- **Online Learning** — model continuously fine-tunes on live detections
- **Incident Alerts** — fires on high density or rapid traffic spikes


## Requirements

```
numpy<2  torch  scikit-learn  pykalman  pandas
opencv-python  ultralytics  streamlit  plotly  yt-dlp  pyyaml
```